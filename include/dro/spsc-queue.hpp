// Andrew Drogalis Copyright (c) 2024, GNU 3.0 Licence
//
// Inspired from Erik Rigtorp
// Significant Modifications / Improvements
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

#ifndef DRO_SPSC_QUEUE
#define DRO_SPSC_QUEUE

#include <atomic>     // for atomic, memory_order
#include <concepts>   // for concept, requires
#include <cstddef>    // for size_t
#include <limits>     // for numeric_limits
#include <new>        // for std::hardware_destructive_interference_size
#include <stdexcept>  // for std::logic_error
#include <type_traits>// for std::is_default_constructible
#include <utility>    // for forward
#include <vector>     // for vector, allocator

namespace dro
{

#ifdef __cpp_lib_hardware_interference_size
static constexpr std::size_t cacheLineSize =
    std::hardware_destructive_interference_size;
#else
static constexpr std::size_t cacheLineSize = 64;
#endif

template <typename T>
concept SPSC_Type =
    std::is_default_constructible<T>::value &&
    std::is_nothrow_destructible<T>::value &&
    (std::is_move_assignable_v<T> || std::is_copy_assignable_v<T>);

template <typename T, typename... Args>
concept SPSC_NoThrow_Type =
    std::is_nothrow_constructible_v<T, Args&&...> &&
    ((std::is_nothrow_copy_assignable_v<T> && std::is_copy_assignable_v<T>) ||
     (std::is_nothrow_move_assignable_v<T> && std::is_move_assignable_v<T>));

template <SPSC_Type T, typename Allocator = std::allocator<T>> class SPSCQueue
{
private:
  std::size_t capacity_;
  std::vector<T, Allocator> buffer_;

  static constexpr std::size_t padding = ((cacheLineSize - 1) / sizeof(T)) + 1;
  static constexpr std::size_t MAX_SIZE_T =
      std::numeric_limits<std::size_t>::max();

  alignas(cacheLineSize) std::atomic<std::size_t> readIndex_ {0};
  alignas(cacheLineSize) std::size_t readIndexCache_ {0};
  alignas(cacheLineSize) std::atomic<std::size_t> writeIndex_ {0};
  alignas(cacheLineSize) std::size_t writeIndexCache_ {0};

public:
  explicit SPSCQueue(const std::size_t capacity,
                     const Allocator& allocator = Allocator())
      : capacity_(capacity), buffer_(allocator)
  {
    // Capacity cannot be negative
    if (capacity_ < 1)
    {
      throw std::logic_error("SPSCQueue capacity must be positive size type");
    }
    // Padding is for preventing cache contention between adjacent memory
    // - 1 is for the ++capacity_ argument (rare overflow edge case)
    capacity_ = (capacity_ < MAX_SIZE_T - 2 * padding - 1)
                    ? capacity_
                    : MAX_SIZE_T - 2 * padding - 1;
    ++capacity_;// prevents live lock e.g. reader and writer share 1 slot for
                // size 1
    buffer_.resize(capacity_ + 2 * padding);
  }

  ~SPSCQueue() = default;
  // non-copyable and non-movable
  SPSCQueue(const SPSCQueue& lhs)            = delete;
  SPSCQueue& operator=(const SPSCQueue& lhs) = delete;
  SPSCQueue(SPSCQueue&& lhs)                 = delete;
  SPSCQueue& operator=(SPSCQueue&& lhs)      = delete;

  template <typename... Args>
    requires std::constructible_from<T, Args...>
  void emplace(Args&&... args) noexcept(SPSC_NoThrow_Type<T, Args...>)
  {
    auto const writeIndex = writeIndex_.load(std::memory_order_relaxed);
    auto nextWriteIndex   = (writeIndex == capacity_ - 1) ? 0 : writeIndex + 1;
    while (nextWriteIndex == readIndexCache_)
    {
      readIndexCache_ = readIndex_.load(std::memory_order_acquire);
    }
    write_value(writeIndex, std::forward<Args...>((args)...));
    writeIndex_.store(nextWriteIndex, std::memory_order_release);
  }

  template <typename... Args>
    requires std::constructible_from<T, Args...>
  void force_emplace(Args&&... args) noexcept(SPSC_NoThrow_Type<T, Args...>)
  {
    auto const writeIndex = writeIndex_.load(std::memory_order_relaxed);
    auto nextWriteIndex   = (writeIndex == capacity_ - 1) ? 0 : writeIndex + 1;
    write_value(writeIndex, std::forward<Args...>((args)...));
    writeIndex_.store(nextWriteIndex, std::memory_order_release);
  }

  template <typename... Args>
    requires std::constructible_from<T, Args...>
  bool try_emplace(Args&&... args) noexcept(SPSC_NoThrow_Type<T, Args...>)
  {
    auto const writeIndex = writeIndex_.load(std::memory_order_relaxed);
    auto nextWriteIndex   = (writeIndex == capacity_ - 1) ? 0 : writeIndex + 1;
    if (nextWriteIndex == readIndexCache_)
    {
      readIndexCache_ = readIndex_.load(std::memory_order_acquire);
      if (nextWriteIndex == readIndexCache_)
      {
        return false;
      }
    }
    write_value(writeIndex, std::forward<Args&&...>((args)...));
    writeIndex_.store(nextWriteIndex, std::memory_order_release);
    return true;
  }

  void push(const T& val) noexcept(SPSC_NoThrow_Type<T>) { emplace(val); }

  template <typename P>
    requires std::constructible_from<T, P&&>
  void push(P&& val) noexcept(SPSC_NoThrow_Type<T, P&&>)
  {
    emplace(std::forward<P>(val));
  }

  void force_push(const T& val) noexcept(SPSC_NoThrow_Type<T>) { force_emplace(val); }

  template <typename P>
    requires std::constructible_from<T, P&&>
  void force_push(P&& val) noexcept(SPSC_NoThrow_Type<T, P&&>)
  {
    force_emplace(std::forward<P>(val));
  }

  [[nodiscard]] bool try_push(const T& val) noexcept(SPSC_NoThrow_Type<T>)
  {
    return try_emplace(val);
  }

  template <typename P>
    requires std::constructible_from<T, P&&>
  [[nodiscard]] bool try_push(P&& val) noexcept(SPSC_NoThrow_Type<T, P&&>)
  {
    return try_emplace(std::forward<P>(val));
  }

  [[nodiscard]] bool try_pop(T& val) noexcept
  {
    const auto readIndex = readIndex_.load(std::memory_order_relaxed);
    if (readIndex == writeIndexCache_)
    {
      writeIndexCache_ = writeIndex_.load(std::memory_order_acquire);
      if (readIndex == writeIndexCache_)
      {
        return false;
      }
    }
    val                = read_value(readIndex);
    auto nextReadIndex = (readIndex == capacity_ - 1) ? 0 : readIndex + 1;
    readIndex_.store(nextReadIndex, std::memory_order_release);
    return true;
  }

  [[nodiscard]] std::size_t size() const noexcept
  {
    auto writeIndex = writeIndex_.load(std::memory_order_acquire);
    auto readIndex  = readIndex_.load(std::memory_order_acquire);
    // This method prevents conversion to std::ptrdiff_t (a signed type)
    if (writeIndex >= readIndex)
    {
      return writeIndex - readIndex;
    }
    return (capacity_ - readIndex) + writeIndex;
  }

  [[nodiscard]] bool empty() const noexcept
  {
    return writeIndex_.load(std::memory_order_acquire) ==
           readIndex_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_ - 1; }

private:
  T& read_value(const auto readIndex) noexcept(SPSC_NoThrow_Type<T>)
    requires std::is_copy_assignable_v<T> && (! std::is_move_assignable_v<T>)
  {
    return buffer_[readIndex + padding];
  }

  T&& read_value(const auto readIndex) noexcept(SPSC_NoThrow_Type<T>)
    requires std::is_move_assignable_v<T>
  {
    return std::move(buffer_[readIndex + padding]);
  }

  void write_value(const auto writeIndex, T val) noexcept(SPSC_NoThrow_Type<T>)
    requires std::is_copy_assignable_v<T> && (! std::is_move_assignable_v<T>)
  {
    buffer_[writeIndex + padding] = val;
  }

  void write_value(const auto writeIndex, T val) noexcept(SPSC_NoThrow_Type<T>)
    requires std::is_move_assignable_v<T>
  {
    buffer_[writeIndex + padding] = std::move(val);
  }

  template <typename... Args>
    requires std::constructible_from<T, Args...> &&
             std::is_copy_assignable_v<T> && (! std::is_move_assignable_v<T>)
  void write_value(const auto writeIndex,
                   Args&&... args) noexcept(SPSC_NoThrow_Type<T, Args...>)
  {
    buffer_[writeIndex + padding] = T(std::forward<Args>(args)...);
  }

  template <typename... Args>
    requires std::constructible_from<T, Args...> && std::is_move_assignable_v<T>
  void write_value(const auto writeIndex,
                   Args&&... args) noexcept(SPSC_NoThrow_Type<T, Args...>)
  {
    buffer_[writeIndex + padding] = std::move(T(std::forward<Args>(args)...));
  }
};

}// namespace dro

#endif
