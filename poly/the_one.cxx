#include <new>
#include <vector>
#include <iostream>

#pragma omp declare target
class Surface {
public:
    virtual double doOne(double x) = 0;
    virtual Surface* getDevPtr() = 0;
};


class S1 : public Surface {
public:
    S1(double c) : _c(c), _devPtr(nullptr) {}

    virtual double doOne(double x) override {
        return x + _c;
    }

    virtual Surface* getDevPtr() override {
        if (_devPtr)
            return _devPtr;
        S1* ptr;
        #pragma omp target map(from:ptr)
        {
            ptr = new S1(_c);
        }
        _devPtr = ptr;
        return ptr;
    }

private:
    double _c;
    Surface* _devPtr;
};

class Sum : public Surface {
public:
    Sum(Surface** surfaces, size_t nsurf) : _surfaces(surfaces), _nsurf(nsurf), _devPtr(nullptr) {
        // Allocate own memory for surfaces
        _surfaces = new Surface*[_nsurf];
        for (int i=0; i<_nsurf; i++) {
            _surfaces[i] = surfaces[i];
        }
    }

    virtual double doOne(double x) override {
        double result = 0;
        for (int i=0; i<_nsurf; i++) {
            result += _surfaces[i]->doOne(x);
        }
        return result;
    }

    virtual Surface* getDevPtr() override {
        if (_devPtr)
            return _devPtr;
        Surface** _devPtrs = new Surface*[_nsurf];
        for(int i=0; i<_nsurf; i++) {
            _devPtrs[i] = _surfaces[i]->getDevPtr();
        }
        Sum* ptr;
        #pragma omp target map(from:ptr) map(to:_devPtrs[:_nsurf])
        {
            ptr = new Sum(_devPtrs, _nsurf);
        }
        _devPtr = ptr;
        return ptr;
    }

private:
    Surface** _surfaces;
    size_t _nsurf;
    Surface* _devPtr;
};
#pragma omp end declare target

int main() {
    S1 s1(1);  // 1 + x
    S1 s2(2);  // 2 + x
    S1 s3(3);  // 3 + x

    Surface* summands[3];
    summands[0] = &s1;
    summands[1] = &s2;
    summands[2] = &s3;
    Sum sum(summands, 3);  // 6 + 3x
    Surface* devPtr = sum.getDevPtr();

    std::vector<double> in(10, 0.0);
    for(int i=0; i<10; i++) {
        in[i] = i;
    }

    std::vector<double> out(10, 0.0);

    double* inptr = in.data();
    double* outptr = out.data();

    #pragma omp target teams distribute parallel for map(inptr[:10], outptr[:10]) is_device_ptr(devPtr)
    for(int i=0; i<10; i++) {
        outptr[i] = devPtr->doOne(inptr[i]);
    }

    for(int i=0; i<10; i++) {
        std::cout << out[i] << '\n';
    }
}
