#include <iostream>
#include <memory>
#include <vector>

#include <omp.h>

// Explicitly make Base and Derived3 class declarations available on the device
// This is needed if a compiler is only OpenMP-4.5 compliant.
// It is no longer needed in an OpenMP-5.0 compliant compiler.
#pragma omp declare target
class Surface {
public:
  virtual double sag(double x, double y) = 0;
  virtual ~Surface() {}
};

class S1 : public Surface {
public:
  double sag(double x, double y) { return x + y; }
};

class S2 : public Surface {
public:
  double sag(double x, double y) { return x * y; }
};

class Sum : public Surface {
public:
  double sag(double x, double y) {
    double result = 0.0;
    for (auto &surface : _surfaces) {
      result += surface->sag(x, y);
    }
    return result;
  }
  void setSurface(int i, Surface *sptr) {
    _surfaces[i] = sptr;
  }

private:
  Surface* _surfaces[2];
};
#pragma omp end declare target

// Create a pointer to the Base and Derived3 object on the device
// This can then be used in multiple target regions
#pragma omp declare target
S1 *d_s1ptr;
S2 *d_s2ptr;
Sum *d_sumptr;
#pragma omp end declare target

int main() {
  S1 s1;
  S2 s2;
  auto s1ptr = std::make_shared<S1>(s1);
  auto s2ptr = std::make_shared<S2>(s2);
  std::vector<std::shared_ptr<Surface>> surfaces({s1ptr, s2ptr});
  Sum sum;
  sum.setSurface(0, static_cast<Surface *>(s1ptr.get()));
  sum.setSurface(1, static_cast<Surface *>(s2ptr.get()));
#pragma omp target
  {
    d_s1ptr = new S1;
    d_s2ptr = new S2;
    d_sumptr = new Sum;
    d_sumptr->setSurface(0, d_s1ptr);
    d_sumptr->setSurface(1, d_s2ptr);
  }

  // Scalar version on CPU
  std::cout << "s1.sag(1, 1) = " << s1.sag(1, 1) << '\n';
  std::cout << "s2.sag(1, 1) = " << s2.sag(1, 1) << '\n';
  std::cout << "sum.sag(1, 1) = " << sum.sag(1, 1) << '\n';

  // Try vector version on GPU
  std::vector<double> x(10000000, 1.0);
  std::vector<double> y(10000000, 1.0);
  std::vector<double> result(10000000, 0.0);

  auto xptr = x.data();
  auto yptr = y.data();
  auto rptr = result.data();

#pragma omp target map(xptr[:10000000], yptr[:10000000], rptr[:10000000])
  {
#pragma omp teams distribute parallel for
    {
      for (int i = 0; i < 10000000; i++) {
        rptr[i] = d_sumptr->sag(xptr[i], yptr[i]);
      }
    }
  }

  std::cout << "GPU output: " << result[0] << "\n";

  return 0;
}
