#include <iostream>
#include <string>
#include <sstream>
#include <chrono>

#include <occa.hpp>

//---[ Internal Tools ]-----------------
// Note: These headers are not officially supported
//       Please don't rely on it outside of the occa examples
//##include </grand/catalyst/spatel/OCCA_ML/occa/src/occa/internal/utils/cli.hpp>
//##include </grand/catalyst/spatel/OCCA_ML/occa/src/occa/internal/utils/testing.hpp>
#include <occa/internal/utils/cli.hpp>
#include <occa/internal/utils/testing.hpp>
//======================================
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <cuda.h>

constexpr int NX = 1<<20; // number of points in spatial discretization

occa::json parseArgs(int argc, const char **argv);
void PyIt(PyObject *p_func, double *u);

int main(int argc, const char **argv) {

  // Some python initialization
  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\".\")");	
  std::cout << "Initialization of Python 1: Done" << std::endl;

  std::cout << "Initializing numpy library" << std::endl;
  // initialize numpy array library
  import_array1(-1);

  std::cout << "Loading python module" << std::endl;
  PyObject* pName = PyUnicode_DecodeFSDefault("python_module"); // Python filename
  PyObject* pModule = PyImport_Import(pName);
  Py_DECREF(pName); // finished with this string so release reference
  std::cout << "Loaded python module" << std::endl;

  std::cout << "Loading functions from module" << std::endl;
  PyObject* pmy_func1 = PyObject_GetAttrString(pModule, "my_function1");
  PyObject* pmy_func2 = PyObject_GetAttrString(pModule, "my_function2");
  Py_DECREF(pModule); // finished with this module so release reference
  std::cout << "Loaded functions" << std::endl;

  occa::json args = parseArgs(argc, argv);

  int entries = NX;

  double *a  = new double[entries];
  double *b  = new double[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = 1.0;
    b[i]  = 0;
  }

  occa::device device;
  occa::memory o_a, o_b;

  //---[ Device Setup ]-------------------------------------
  device.setup((std::string) args["options/device"]);
  device.setup({
     {"mode"     , "CUDA"},
     {"device_id", 0},
   });

  // Allocate memory on the device
  o_a = device.malloc<double>(entries);
  o_b = device.malloc<double>(entries);
  
  //Get Backend Pointer
  double *d_b = static_cast<double *>(o_b.ptr());

  // Compile the kernel at run-time
  occa::kernel copyVectors = device.buildKernel("copyVectors.okl","copyVectors");
  
  // Copy memory to the device
  o_a.copyFrom(a);
  o_b.copyFrom(b);

  // Launch device kernel
  copyVectors(entries, o_a, o_b);

  int Ntests=1000;
  // D->H->D
  auto walltime_start = std::chrono::high_resolution_clock::now();
  for(int test = 0; test < Ntests; ++test) {

      // Run the OCCA kernel
      copyVectors(entries, o_a, o_b);

      // Copy data Host
      o_b.copyTo(b);

      // Pass data through Python-Interpreter
      PyIt(pmy_func1, b);      

  }
  auto walltime_finish = std::chrono::high_resolution_clock::now();
  double wallTime = std::chrono::duration<double,std::milli>(walltime_finish-walltime_start).count(); 
  std::cout << "DHD total mean wallTime : " << wallTime/Ntests << std::endl;
  
  // Assert values
  for (int i = 0; i < entries; ++i) {
    if (!occa::areBitwiseEqual(b[i], a[i])) {
      throw 1;
    }
  }
  
  walltime_start = std::chrono::high_resolution_clock::now();
  for(int test = 0; test < Ntests; ++test) {

      // Run the OCCA kernel
      copyVectors(entries, o_a, o_b);

      // Pass data through Python-Interpreter
      PyIt(pmy_func2, d_b);      

  }
  walltime_finish = std::chrono::high_resolution_clock::now();
  wallTime = std::chrono::duration<double,std::milli>(walltime_finish-walltime_start).count(); 
  std::cout << "DD total mean wallTime : " << wallTime/Ntests << std::endl;

  o_b.copyTo(b);
  // Assert values
  for (int i = 0; i < entries; ++i) {
    if (!occa::areBitwiseEqual(b[i], 50*a[i])) {
      throw 1;
    }
  }

  // Free host memory
  delete [] a;
  delete [] b;

  return 0;
}

occa::json parseArgs(int argc, const char **argv) {
  occa::cli::parser parser;
  parser
    .withDescription(
      "Example adding two vectors"
    )
    .addOption(
      occa::cli::option('d', "device",
                        "Device properties (default: \"{mode: 'Serial'}\")")
      .withArg()
      .withDefaultValue("{mode: 'Serial'}")
    )
    .addOption(
      occa::cli::option('v', "verbose",
                        "Compile kernels in verbose mode")
    );

  occa::json args = parser.parseArgs(argc, argv);
  occa::settings()["kernel/verbose"] = args["options/verbose"];

  return args;
}

void PyIt(PyObject *p_func, double *u)
{
  PyObject* pArgs = PyTuple_New(1);

  //Numpy array dimensions
  npy_intp dim[] = {NX};

  // create a new Python array that is a wrapper around u (not a copy) and put it in tuple pArgs
  PyObject* array_1d = PyArray_SimpleNewFromData(1, dim, NPY_FLOAT64, u);
  PyTuple_SetItem(pArgs, 0, array_1d);

  // pass array into our Python function and cast result to PyArrayObject
  PyArrayObject* pValue = (PyArrayObject*) PyObject_CallObject(p_func, pArgs);
  //std::cout << "Called python data collection function successfully"<<std::endl;

  Py_DECREF(pArgs);
  Py_DECREF(pValue);
  // We don't need to decref array_1d because PyTuple_SetItem steals a reference
}
