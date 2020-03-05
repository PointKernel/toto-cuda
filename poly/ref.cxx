#include <array>
#include <iostream>
#include <omp.h>
#include <vector>

// Explicitly make Base and Derived3 class declarations available on the device
// This is needed if a compiler is only OpenMP-4.5 compliant.
// It is no longer needed in an OpenMP-5.0 compliant compiler.
#pragma omp declare target
class Base {
public:
  Base() = default;
  Base(double a) : _a(a) {}
  virtual double doOne(double x) const { return x * _a + 3 * _a; }
  virtual size_t cloneOntoDevice() const { return sizeof(_a); }

private:
  double _a;
};

class Derived3 : public Base {
public:
  Derived3(double c) : _c(c) {}
  virtual double doOne(double x) const override { return x * _c + 3 * _c; }
  virtual size_t cloneOntoDevice() const override { return 0; }

private:
  double _c;
};
#pragma omp end declare target

// Create a pointer to the Base and Derived3 object on the device
// This can then be used in multiple target regions
#pragma omp declare target
Base *gb;
Derived3 *gd3;
#pragma omp end declare target

int main() {
  std::vector<double> in(10);
  std::vector<double> out(10, 0.0);
  for (int i = 0; i < 10; i++)
    in[i] = i;
  for (int i = 0; i < 10; i++)
    std::cout << out[i] << ' ';
  std::cout << '\n';
  double *inptr = in.data();
  double *outptr = out.data();

  // Create Base gb and Derived gd3 directly on device.
#pragma omp target
  {
    gb = new Base(5.0);
    gd3 = new Derived3(3.0);
  }
#pragma omp target
  { gb = new Base(3.0); }

  // Access the device version of gb and gd3
  // Call its doOne method via a Base pointer.
#pragma omp target map(inptr [0:10]) map(outptr [0:10])
  {
#pragma omp teams distribute parallel for
    {
      for (int i = 0; i < 10; i++) {
        Base *p = gb;
        outptr[i] = p->doOne(inptr[i]); // This vtable lookup works.
      }
    }
  }
#pragma omp target map(inptr [0:10]) map(outptr [0:10])
  {
#pragma omp teams distribute parallel for
    {
      for (int i = 0; i < 10; i++) {
        Base *p = gd3;
        outptr[i] += p->doOne(inptr[i]); // This vtable lookup works.
      }
    }
  }

  for (int i = 0; i < 10; i++)
    std::cout << out[i] << ' ';
  std::cout << '\n';
  return 0;
}
