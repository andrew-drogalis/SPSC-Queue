// Copyright (c) 2024 Andrew Drogalis
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the “Software”), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

#ifndef DRO_SPSC_QUEUE
#define DRO_SPSC_QUEUE

#include <array>       // for std::array
#include <atomic>      // for atomic, memory_order
#include <concepts>    // for concept, requires
#include <cstddef>     // for size_t
#include <limits>      // for numeric_limits
#include <new>         // for std::hardware_destructive_interference_size
#include <stdexcept>   // for std::logic_error
#include <type_traits> // for std::is_default_constructible
#include <utility>     // for forward
#include <vector>      // for vector, allocator

namespace dro {

namespace details {

#ifdef __cpp_lib_hardware_interference_size
static constexpr std::size_t cacheLineSize =
    std::hardware_destructive_interference_size;
#else
static constexpr std::size_t cacheLineSize = 64;
#endif

static constexpr std::size_t MAX_BYTES_ON_STACK = 2'097'152; // 2 MBs

template <typename T>
concept SPSC_Type =
    std::is_default_constructible_v<T> && std::is_nothrow_destructible_v<T> &&
    (std::is_move_assignable_v<T> || std::is_copy_assignable_v<T>);

template <typename T, typename... Args>
concept SPSC_NoThrow_Type =
    std::is_nothrow_constructible_v<T, Args &&...> &&
    ((std::is_nothrow_copy_assignable_v<T> && std::is_copy_assignable_v<T>) ||
     (std::is_nothrow_move_assignable_v<T> && std::is_move_assignable_v<T>));

// Prevents Stack Overflow
template <typename T, std::size_t N>
concept MAX_STACK_SIZE = (N <= (MAX_BYTES_ON_STACK / sizeof(T)));

// Memory Allocated on the Heap (Default Option)
template <SPSC_Type T, typename Allocator = std::allocator<T>>
struct HeapBuffer {
  const std::size_t capacity_;
  std::vector<T, Allocator> buffer_;

  static constexpr std::size_t padding = ((cacheLineSize - 1) / sizeof(T)) + 1;
  static constexpr std::size_t MAX_SIZE_T =
      std::numeric_limits<std::size_t>::max();

  explicit HeapBuffer(const std::size_t capacity,
                      const Allocator &allocator = Allocator())
      // +1 prevents live lock e.g. reader and writer share 1 slot for size 1
      : capacity_(capacity + 1), buffer_(allocator) {
    if (capacity < 1) {
      throw std::logic_error("Capacity must be a positive number; Heap "
                             "allocations require capacity argument");
    }
    // (2 * padding) is for preventing cache contention between adjacent memory
    if (capacity_ > MAX_SIZE_T - (2 * padding)) {
      throw std::overflow_error(
          "Capacity with padding exceeds std::size_t. Reduce size of queue.");
    }
    buffer_.resize(capacity_ + (2 * padding));
  }

  ~HeapBuffer() = default;
  // Non-Copyable and Non-Movable
  HeapBuffer(const HeapBuffer &lhs) = delete;
  HeapBuffer &operator=(const HeapBuffer &lhs) = delete;
  HeapBuffer(HeapBuffer &&lhs) = delete;
  HeapBuffer &operator=(HeapBuffer &&lhs) = delete;
};

// Memory Allocated on the Stack
template <SPSC_Type T, std::size_t N, typename Allocator = std::allocator<T>>
struct StackBuffer {
  // +1 prevents live lock e.g. reader and writer share 1 slot for size 1
  static constexpr std::size_t capacity_{N + 1};
  static constexpr std::size_t padding = ((cacheLineSize - 1) / sizeof(T)) + 1;
  // (2 * padding) is for preventing cache contention between adjacent memory
  std::array<T, capacity_ + (2 * padding)> buffer_;

  explicit StackBuffer(const std::size_t capacity,
                       [[maybe_unused]] const Allocator &allocator = Allocator()) {
    if (capacity) {
      throw std::invalid_argument(
          "Capacity in constructor is ignored for stack allocations");
    }
  }

  ~StackBuffer() = default;
  // Non-Copyable and Non-Movable
  StackBuffer(const StackBuffer &lhs) = delete;
  StackBuffer &operator=(const StackBuffer &lhs) = delete;
  StackBuffer(StackBuffer &&lhs) = delete;
  StackBuffer &operator=(StackBuffer &&lhs) = delete;
};

} // namespace details

template <details::SPSC_Type T, std::size_t N = 0,
          typename Allocator = std::allocator<T>>
  requires details::MAX_STACK_SIZE<T, N>
class SPSCQueue
    : public std::conditional_t<N == 0, details::HeapBuffer<T, Allocator>,
                                details::StackBuffer<T, N>> {
private:
  using base_type =
      std::conditional_t<N == 0, details::HeapBuffer<T, Allocator>,
                         details::StackBuffer<T, N>>;
  static constexpr bool nothrow_v = details::SPSC_NoThrow_Type<T>;

  struct alignas(details::cacheLineSize) WriterCacheLine {
    std::atomic<std::size_t> writeIndex_{0};
    std::size_t readIndexCache_{0};
    // Reduces cache contention on very small queues
    const size_t paddingCache_ = base_type::padding;
  } writer_;

  struct alignas(details::cacheLineSize) ReaderCacheLine {
    std::atomic<std::size_t> readIndex_{0};
    std::size_t writeIndexCache_{0};
    // Reduces cache contention on very small queues
    std::size_t capacityCache_{};
  } reader_;

public:
  explicit SPSCQueue(const std::size_t capacity = 0,
                     const Allocator &allocator = Allocator())
      : base_type(capacity, allocator) {
    reader_.capacityCache_ = base_type::capacity_;
  }

  ~SPSCQueue() = default;
  // Non-Copyable and Non-Movable
  SPSCQueue(const SPSCQueue &lhs) = delete;
  SPSCQueue &operator=(const SPSCQueue &lhs) = delete;
  SPSCQueue(SPSCQueue &&lhs) = delete;
  SPSCQueue &operator=(SPSCQueue &&lhs) = delete;

  template <typename... Args>
    requires std::constructible_from<T, Args &&...>
  void
  emplace(Args &&...args) noexcept(details::SPSC_NoThrow_Type<T, Args &&...>) {
    const auto writeIndex = writer_.writeIndex_.load(std::memory_order_relaxed);
    const auto nextWriteIndex =
        (writeIndex == base_type::capacity_ - 1) ? 0 : writeIndex + 1;

    // Loop while waiting for reader to catch up
    while (nextWriteIndex == writer_.readIndexCache_) {
      writer_.readIndexCache_ =
          reader_.readIndex_.load(std::memory_order_acquire);
    }
    write_value(writeIndex, std::forward<Args>(args)...);
    writer_.writeIndex_.store(nextWriteIndex, std::memory_order_release);
  }

  template <typename... Args>
    requires std::constructible_from<T, Args &&...>
  void force_emplace(Args &&...args) noexcept(
      details::SPSC_NoThrow_Type<T, Args &&...>) {
    const auto writeIndex = writer_.writeIndex_.load(std::memory_order_relaxed);
    const auto nextWriteIndex =
        (writeIndex == base_type::capacity_ - 1) ? 0 : writeIndex + 1;

    write_value(writeIndex, std::forward<Args>(args)...);
    writer_.writeIndex_.store(nextWriteIndex, std::memory_order_release);
  }

  template <typename... Args>
    requires std::constructible_from<T, Args &&...>
  [[nodiscard]] bool try_emplace(Args &&...args) noexcept(
      details::SPSC_NoThrow_Type<T, Args &&...>) {
    const auto writeIndex = writer_.writeIndex_.load(std::memory_order_relaxed);
    const auto nextWriteIndex =
        (writeIndex == base_type::capacity_ - 1) ? 0 : writeIndex + 1;

    // Check reader cache and if actually equal then fail to write
    if (nextWriteIndex == writer_.readIndexCache_) {
      writer_.readIndexCache_ =
          reader_.readIndex_.load(std::memory_order_acquire);
      if (nextWriteIndex == writer_.readIndexCache_) {
        return false;
      }
    }
    write_value(writeIndex, std::forward<Args>(args)...);
    writer_.writeIndex_.store(nextWriteIndex, std::memory_order_release);
    return true;
  }

  void push(const T &val) noexcept(nothrow_v) { emplace(val); }

  template <typename P>
    requires std::constructible_from<T, P &&>
  void push(P &&val) noexcept(details::SPSC_NoThrow_Type<T, P &&>) {
    emplace(std::forward<P>(val));
  }

  void force_push(const T &val) noexcept(nothrow_v) { force_emplace(val); }

  template <typename P>
    requires std::constructible_from<T, P &&>
  void force_push(P &&val) noexcept(details::SPSC_NoThrow_Type<T, P &&>) {
    force_emplace(std::forward<P>(val));
  }

  [[nodiscard]] bool try_push(const T &val) noexcept(nothrow_v) {
    return try_emplace(val);
  }

  template <typename P>
    requires std::constructible_from<T, P &&>
  [[nodiscard]] bool
  try_push(P &&val) noexcept(details::SPSC_NoThrow_Type<T, P &&>) {
    return try_emplace(std::forward<P>(val));
  }

  void pop(T &val) noexcept(nothrow_v) {
    const auto readIndex = reader_.readIndex_.load(std::memory_order_relaxed);

    // Loop while waiting for writer to enqueue
    while (readIndex == reader_.writeIndexCache_) {
      reader_.writeIndexCache_ =
          writer_.writeIndex_.load(std::memory_order_acquire);
    }
    read_value(readIndex, val);

    const auto nextReadIndex =
        (readIndex == reader_.capacityCache_ - 1) ? 0 : readIndex + 1;
    reader_.readIndex_.store(nextReadIndex, std::memory_order_release);
  }

  [[nodiscard]] bool try_pop(T &val) noexcept(nothrow_v) {
    const auto readIndex = reader_.readIndex_.load(std::memory_order_relaxed);

    // Check writer cache and if actually equal then fail to read
    if (readIndex == reader_.writeIndexCache_) {
      reader_.writeIndexCache_ =
          writer_.writeIndex_.load(std::memory_order_acquire);
      if (readIndex == reader_.writeIndexCache_) {
        return false;
      }
    }

    read_value(readIndex, val);
    const auto nextReadIndex =
        (readIndex == reader_.capacityCache_ - 1) ? 0 : readIndex + 1;
    reader_.readIndex_.store(nextReadIndex, std::memory_order_release);
    return true;
  }

  [[nodiscard]] std::size_t size() const noexcept {
    const auto writeIndex = writer_.writeIndex_.load(std::memory_order_acquire);
    const auto readIndex = reader_.readIndex_.load(std::memory_order_acquire);

    // This method prevents conversion to std::ptrdiff_t (a signed type)
    if (writeIndex >= readIndex) {
      return writeIndex - readIndex;
    }
    return (base_type::capacity_ - readIndex) + writeIndex;
  }

  [[nodiscard]] bool empty() const noexcept {
    return writer_.writeIndex_.load(std::memory_order_acquire) ==
           reader_.readIndex_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::size_t capacity() const noexcept {
    return base_type::capacity_ - 1;
  }

private:
  // Note: The "+ padding" is a constant offset used to prevent false sharing
  // with memory in front of the SPSC allocations
  void read_value(std::size_t readIndex, T &val) noexcept(nothrow_v) {
    if constexpr (std::is_move_assignable_v<T>) {
      val = std::move(base_type::buffer_[readIndex + base_type::padding]);
    } else {
      val = base_type::buffer_[readIndex + base_type::padding];
    }
  }

  template <typename U>
    requires(std::same_as<T, U>)
  void write_value(std::size_t writeIndex, U &&val) noexcept(nothrow_v) {
    base_type::buffer_[writeIndex + writer_.paddingCache_] =
        std::forward<U>(val);
  }

  template <typename... Args>
    requires(std::constructible_from<T, Args && ...>)
  void write_value(std::size_t writeIndex, Args &&...args) noexcept(
      details::SPSC_NoThrow_Type<T, Args &&...>) {
    if constexpr (std::is_move_assignable_v<T>) {
      base_type::buffer_[writeIndex + writer_.paddingCache_] =
          T(std::forward<Args>(args)...);
    } else {
      T copyOnly{std::forward<Args>(args)...};
      base_type::buffer_[writeIndex + writer_.paddingCache_] = copyOnly;
    }
  }
};

} // namespace dro
#endif
