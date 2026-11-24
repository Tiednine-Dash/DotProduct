/****************************************************************************************/
// System includes
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <span>
#include <vector>

/****************************************************************************************/
// Function Declarations
float
dotProductLibrary (std::span<float const> a, std::span<float const> b);

float
dotProductSimd (std::span<float const> a, std::span<float const> b);

/****************************************************************************************/
// Functions
template<typename T>
  requires std::is_arithmetic_v<T>
bool
equal (T a, T b, size_t N, double relativeError = 1e-4)
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
// Test cases
TEST_CASE ("Basic correctness")
{
  std::vector<float> a { 1, 2, 3, 4 };
  std::vector<float> b { 5, 6, 7, 8 };

  float lib = dotProductLibrary (a, b);
  float simd = dotProductSimd (a, b);

  CHECK (equal (lib, simd, a.size ()));
}

TEST_CASE ("All zeros")
{
  std::vector<float> a (100, 0.0f);
  std::vector<float> b (100, 0.0f);

  CHECK (dotProductSimd (a, b) == 0.0f);
}

TEST_CASE ("Negative values")
{
  std::vector<float> a { -1, -2, -3 };
  std::vector<float> b { 4, 5, 6 };

  float lib = dotProductLibrary (a, b);
  float simd = dotProductSimd (a, b);

  CHECK (equal (lib, simd, a.size ()));
}

TEST_CASE ("Non-multiple of SIMD width")
{
  std::vector<float> a { 1, 2, 3, 4, 5, 6, 7 };
  std::vector<float> b { 7, 6, 5, 4, 3, 2, 1 };

  float lib = dotProductLibrary (a, b);
  float simd = dotProductSimd (a, b);

  CHECK (equal (lib, simd, a.size ()));
}

TEST_CASE ("Large input")
{
  const size_t N = 100000;

  std::vector<float> a (N);
  std::vector<float> b (N);

  for (size_t i = 0; i < N; ++i)
  {
    a[i] = static_cast<float> (i % 100);
    b[i] = static_cast<float> ((i * 2) % 100);
  }

  float lib = dotProductLibrary (a, b);
  float simd = dotProductSimd (a, b);

  CHECK (equal (lib, simd, N));
}

TEST_CASE ("Single element")
{
  std::vector<float> a { 3.0f };
  std::vector<float> b { 4.0f };

  CHECK (dotProductSimd (a, b) == doctest::Approx (12.0f));
}

TEST_CASE ("Size 0")
{
  std::vector<float> a;
  std::vector<float> b;

  CHECK (dotProductSimd (a, b) == 0.0f);
}

TEST_CASE ("Size 1 zero")
{
  std::vector<float> a { 0.0f };
  std::vector<float> b { 5.0f };

  CHECK (dotProductSimd (a, b) == 0.0f);
}

TEST_CASE ("Size 2")
{
  std::vector<float> a { 1, 2 };
  std::vector<float> b { 3, 4 };

  CHECK (equal (dotProductLibrary (a, b), dotProductSimd (a, b), a.size ()));
}

TEST_CASE ("Exactly 8 elements")
{
  std::vector<float> a (8, 1.0f);
  std::vector<float> b (8, 2.0f);

  CHECK (dotProductSimd (a, b) == doctest::Approx (16.0f));
}

TEST_CASE ("Exactly 32 elements (full unroll)")
{
  std::vector<float> a (32, 1.0f);
  std::vector<float> b (32, 1.0f);

  CHECK (dotProductSimd (a, b) == doctest::Approx (32.0f));
}

TEST_CASE ("33 elements (tail case)")
{
  std::vector<float> a (33, 1.0f);
  std::vector<float> b (33, 1.0f);

  CHECK (dotProductSimd (a, b) == doctest::Approx (33.0f));
}

TEST_CASE ("31 elements (just under full chunk)")
{
  std::vector<float> a (31, 1.0f);
  std::vector<float> b (31, 1.0f);

  CHECK (dotProductSimd (a, b) == doctest::Approx (31.0f));
}

TEST_CASE ("Alternating signs")
{
  std::vector<float> a { 1, -1, 1, -1, 1, -1, 1, -1 };
  std::vector<float> b { 1, 1, 1, 1, 1, 1, 1, 1 };

  CHECK (dotProductSimd (a, b) == 0.0f);
}

TEST_CASE ("Increasing sequence")
{
  std::vector<float> a (1000), b (1000);

  for (int i = 0; i < 1000; ++i)
  {
    a[i] = i;
    b[i] = i;
  }

  CHECK (equal (dotProductLibrary (a, b), dotProductSimd (a, b), a.size ()));
}

TEST_CASE ("Decreasing vs increasing")
{
  std::vector<float> a (1000), b (1000);

  for (int i = 0; i < 1000; ++i)
  {
    a[i] = i;
    b[i] = 1000 - i;
  }

  CHECK (equal (dotProductLibrary (a, b), dotProductSimd (a, b), a.size ()));
}

TEST_CASE ("Small floats")
{
  std::vector<float> a (1000, 1e-6f);
  std::vector<float> b (1000, 1e-6f);

  CHECK (equal (dotProductLibrary (a, b), dotProductSimd (a, b), a.size ()));
}

TEST_CASE ("Large floats")
{
  std::vector<float> a (1000, 1e6f);
  std::vector<float> b (1000, 1e6f);

  CHECK (equal (dotProductLibrary (a, b), dotProductSimd (a, b), a.size ()));
}

TEST_CASE ("Mixed magnitude")
{
  std::vector<float> a (1000), b (1000);

  for (int i = 0; i < 1000; ++i)
  {
    a[i] = (i % 2 == 0) ? 1e6f : 1e-6f;
    b[i] = (i % 2 == 0) ? 1e-6f : 1e6f;
  }

  CHECK (equal (dotProductLibrary (a, b), dotProductSimd (a, b), a.size ()));
}

TEST_CASE ("Pseudo-random pattern")
{
  std::vector<float> a (10000), b (10000);

  for (int i = 0; i < 10000; ++i)
  {
    a[i] = (i * 37) % 100;
    b[i] = (i * 91) % 100;
  }

  CHECK (equal (dotProductLibrary (a, b), dotProductSimd (a, b), a.size ()));
}

TEST_CASE ("Cancellation pattern")
{
  std::vector<float> a { 1e6f, -1e6f, 1e6f, -1e6f };
  std::vector<float> b { 1, 1, 1, 1 };

  CHECK (equal (dotProductLibrary (a, b), dotProductSimd (a, b), a.size ()));
}

TEST_CASE ("Near-zero result")
{
  std::vector<float> a { 1, -1, 1, -1 };
  std::vector<float> b { 1, 1, 1, 1 };

  CHECK (equal (dotProductLibrary (a, b), dotProductSimd (a, b), a.size ()));
}

TEST_CASE ("All same value")
{
  std::vector<float> a (5000, 3.14f);
  std::vector<float> b (5000, 2.71f);

  CHECK (equal (dotProductLibrary (a, b), dotProductSimd (a, b), a.size ()));
}

TEST_CASE ("Large non-multiple size")
{
  size_t N = 100003;

  std::vector<float> a (N, 1.0f);
  std::vector<float> b (N, 2.0f);

  CHECK (dotProductSimd (a, b) == doctest::Approx (2.0f * N));
}

TEST_CASE ("Very large input")
{
  size_t N = 500000;

  std::vector<float> a (N, 1.0f);
  std::vector<float> b (N, 1.0f);

  CHECK (equal (dotProductLibrary (a, b), dotProductSimd (a, b), N));
}
