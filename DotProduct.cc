/****************************************************************************************/
// System includes
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <print>
#include <random>
#include <span>
#include <vector>

/****************************************************************************************/
// Local includes
#include "Timer.hpp"

/****************************************************************************************/
// Using declarations
using std::print;
using std::vector;

/****************************************************************************************/
// Function declarations

template<typename T, typename U>
  requires std::is_arithmetic_v<T>
void
fillRandom (std::span<T> seq, U min, U max, unsigned seed);

float
dotProductLibrary (std::span<float const> a, std::span<float const> b);

float
dotProductSimd (std::span<float const> a, std::span<float const> b);

template<typename ReturnType, typename F>
std::pair<ReturnType, double>
timeReduction (F&& func);

template<typename T>
  requires std::is_arithmetic_v<T>
bool
equal (T a, T b, size_t N, double relativeError = 1e-4);

template<typename T>
struct AlignedAllocator
{
  using value_type = T;
  static constexpr std::size_t ALIGN = 32; // Fixed to 32 for AVX

  AlignedAllocator () noexcept = default;

  template<typename U>
  AlignedAllocator (const AlignedAllocator<U>&) noexcept
  {
  }

  T*
  allocate (std::size_t n)
  {
    if (n == 0)
      return nullptr;

    size_t size = (n * sizeof (T) + ALIGN - 1) & ~(ALIGN - 1);

    if (auto p = static_cast<T*> (std::aligned_alloc (ALIGN, size)))
    {
      return p;
    }
    throw std::bad_alloc ();
  }

  void
  deallocate (T* p, std::size_t) noexcept
  {
    std::free (p);
  }

  // Equality operators are required for full compliance
  bool
  operator== (const AlignedAllocator&) const noexcept
  {
    return true;
  }
  bool
  operator!= (const AlignedAllocator&) const noexcept
  {
    return false;
  }
};

/****************************************************************************************/
#ifndef TESTING
int
main ()
{
  print ("N ==> ");

  std::string input;
  std::cin >> input;
  input.erase (std::remove (input.begin (), input.end (), '\''), input.end ());
  size_t N = std::stoul (input);

  vector<float, AlignedAllocator<float>> vec1 (N), vec2 (N);

  fillRandom (std::span { vec1 }, -4.0f, 4.0f, 1);
  fillRandom (std::span { vec2 }, -4.0f, 4.0f, 1);

  auto [libSum, libTime] =
    timeReduction<float> ([&] () { return dotProductLibrary (vec1, vec2); });

  auto [simdSum, simdTime] =
    timeReduction<float> ([&] () { return dotProductSimd (vec1, vec2); });

  print ("Library: {:.4f}\n", libSum);
  print ("Time:    {:.2f} ms\n\n", libTime);

  print ("SIMD:    {:.4f}\n", simdSum);
  print ("Time:    {:.2f} ms\n", simdTime);

  double speedup = libTime / simdTime;
  print ("Speedup: {:.2f}\n\n", speedup);

  print ("Correct: {}\n", equal (libSum, simdSum, N) ? "yes" : "NO!");
}
#endif
/****************************************************************************************/

template<typename T>
  requires std::is_arithmetic_v<T>
bool
equal (T a, T b, size_t N, double relativeError)
{
  if constexpr (std::floating_point<T>)
  {
    // Handle inf-s, which are problematic when subtracted
    if (a == b)
      return true;
    // Get distance b/w two points
    T diff { std::abs (a - b) };
    // Find largest in absolute value, using "min" for values close to 0
    T max { std::max (
      { std::abs (a), std::abs (b), std::numeric_limits<T>::min () }) };

    return diff < max * std::log2 (N) * relativeError;
  }
  else
    return a == b;
}

/****************************************************************************************/

template<std::size_t N, typename F>
void
unroll (F&& f)
{
  [f]<std::size_t... Is> (std::index_sequence<Is...>)
  {
    // Unrolls into f (0), f (1), ..., f (N - 1)
    (f (std::integral_constant<std::size_t, Is> {}), ...);
  }(std::make_index_sequence<N> {});
}

/**
 * Fills a sequence with random values in a given range.
 *
 * @tparam T Arithmetic type of the sequence elements.
 * @tparam U Type for min/max range values.
 * @param seq The sequence to fill.
 * @param min Minimum random value.
 * @param max Maximum random value.
 * @param seed Seed for reproducibility.
 */
template<typename T, typename U>
  requires std::is_arithmetic_v<T>
void
fillRandom (std::span<T> seq, U min, U max, unsigned seed)
{
  static std::minstd_rand gen (seed);
  std::uniform_real_distribution<float> dist (min, max);
  std::ranges::generate (seq, [&] () { return static_cast<T> (dist (gen)); });
}

/**
 * Computes dot product using the C++ standard library.
 *
 * @param a First vector.
 * @param b Second vector.
 * @return Dot product of a and b.
 */
float
dotProductLibrary (std::span<float const> a, std::span<float const> b)
{
  return std::inner_product (a.begin (), a.end (), b.begin (), 0.0f);
}

/**
 * Computes dot product using AVX SIMD instructions with loop unrolling.
 *
 * Uses 256-bit registers (8 floats per register) and 4 accumulators
 * to reduce latency from fused multiply-add operations.
 *
 * @param a First vector (must be aligned).
 * @param b Second vector (must be aligned).
 * @return Dot product of a and b.
 */
float
dotProductSimd (std::span<float const> a, std::span<float const> b)
{
  constexpr int VEC_WIDTH = 8;
  constexpr int UNROLL_FACTOR = 2;

  const size_t N = a.size ();
  const float* pa = a.data ();
  const float* pb = b.data ();

  constexpr size_t STEP = VEC_WIDTH * UNROLL_FACTOR;

  size_t i = 0;

  __m256 acc[UNROLL_FACTOR];
  for (int k = 0; k < UNROLL_FACTOR; ++k)
    acc[k] = _mm256_setzero_ps ();

  for (; i + STEP <= N; i += STEP)
  {
    unroll<UNROLL_FACTOR> (
      [&] (auto k)
      {
        constexpr int offset = k * VEC_WIDTH;

        acc[k] = _mm256_fmadd_ps (_mm256_loadu_ps (pa + i + offset),
                                  _mm256_loadu_ps (pb + i + offset),
                                  acc[k]);
      });
  }

  __m256 sumV = acc[0];
  for (int k = 1; k < UNROLL_FACTOR; ++k)
    sumV = _mm256_add_ps (sumV, acc[k]);

  __m128 vlow = _mm256_castps256_ps128 (sumV);
  __m128 vhigh = _mm256_extractf128_ps (sumV, 1);
  __m128 s = _mm_add_ps (vlow, vhigh);
  s = _mm_hadd_ps (s, s);
  s = _mm_hadd_ps (s, s);

  float result = _mm_cvtss_f32 (s);

  for (; i < N; ++i)
    result += pa[i] * pb[i];

  return result;
}

/**
 * Measures execution time of a function returning a value.
 *
 * @tparam ReturnType Type of the returned value.
 * @tparam F Callable type.
 * @param func Function to execute.
 * @return Pair of (result, execution time in ms).
 */
template<typename ReturnType, typename F>
std::pair<ReturnType, double>
timeReduction (F&& func)
{
  Timer<> t;

  t.start ();
  ReturnType result = func ();
  t.stop ();

  return { result, t.getElapsedMs () };
}
