#include <iostream>
#include <string>
#include <sstream>

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

occa::json parseArgs(int argc, const char **argv);
long hexadecimalToDecimal(std::string hexVal);

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
  PyObject* pmy_func = PyObject_GetAttrString(pModule, "my_function");

  occa::json args = parseArgs(argc, argv);

  int entries = 12;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occa::device device;
  occa::memory o_a, o_b, o_ab;

  //---[ Device Setup ]-------------------------------------
  device.setup((std::string) args["options/device"]);
  device.setup({
     {"mode"     , "CUDA"},
     {"device_id", 0},
   });

  // Allocate memory on the device
  o_a = device.malloc<float>(entries);
  o_b = device.malloc<float>(entries);

  // We can also allocate memory without a dtype
  // WARNING: This will disable runtime type checking
  o_ab = device.malloc(entries * sizeof(float));

  // Compile the kernel at run-time
  occa::kernel addVectors = device.buildKernel("addVectors.okl","addVectors");
  
  // Copy memory to the device
  o_a.copyFrom(a);
  o_b.copyFrom(b);

  //Use OCCA API for back-end pointer
  std::cout << "OCCA ptr Call 1" << std::endl;
  void *my_o_a;  
  my_o_a = o_a.ptr();
  void *my_o_ab;  
  my_o_ab = o_ab.ptr();
  std::cout << "OCCA ptr Call 2" << std::endl;
 
  //CUdeviceptr *d_a = static_cast<CUdeviceptr*>(o_a.ptr());
  float *d_a  = static_cast<float *>(o_a.ptr());
  float *d_ab = static_cast<float *>(o_ab.ptr());

  // PyObject * pArgs=PyCapsule_New(my_o_a, NULL, NULL);
  // Create Numpy array dimensions and 
  // a new Python array that is a 
  // wrapper around u (not a copy) and put it in tuple pArgs
  PyObject* pArgs = PyTuple_New(1); 
  npy_intp dim[]={entries};
  PyObject* array_1d_a = PyArray_SimpleNewFromData(1, dim, NPY_FLOAT32, d_a);  
  //PyObject * array_1d = PyCapsule_New(my_o_a, NULL, NULL);
  
  int rval = PyTuple_SetItem(pArgs, 0, array_1d_a);
  //if (rval == -1){
  //   std::cout << "PyTuple_SetItem return value : " << rval << '\n';
  //}
  std::cout << "OCCA ptr Call 3: Show o_a" << std::endl;
  PyArrayObject* pValue = (PyArrayObject*) PyObject_CallObject(pmy_func, pArgs);
  //PyObject* pValue = (PyObject*)PyObject_CallObject(pmy_func, pArgs);

  if (pValue!=NULL){
  //      printf("Result of call: %ld\n", PyLong_AsLong(pValue));
  //      Py_DECREF(pValue);
  } 
  else {
	Py_DECREF(pmy_func);
        Py_DECREF(pModule);
	PyErr_Print();
        fprintf(stderr,"Call failed\n");
	return 1;
  }

  // Launch device kernel
  addVectors(entries, o_a, o_b, o_ab);
  
  std::cout << "OCCA ptr Call 4: Show o_ab" << std::endl;
  PyObject* array_1d_ab = PyArray_SimpleNewFromData(1, dim, NPY_FLOAT32, d_ab);  
  rval = PyTuple_SetItem(pArgs, 0, array_1d_ab);
  pValue = (PyArrayObject*) PyObject_CallObject(pmy_func, pArgs);

  // Copy result to the host
  o_ab.copyTo(ab);

  std::ostringstream sd_a, sd_ab;
  sd_a << (void*)d_a;
  sd_ab << (void*)d_ab;

  std::string sa  = sd_a.str();
  std::string sab = sd_ab.str();
  //convert-to-base 10
  unsigned long x_a  = std::stoul(sa, nullptr, 16); 
  unsigned long x_ab = std::stoul(sab, nullptr, 16); 
  std::cout << "Memory Address for d_a  in HexaDecimal: " << sa  << ", in Decimal: " << x_a << '\n';
  std::cout << "Memory Address for d_ab in HexaDecimal: " << sab << ", in Decimal: " << x_ab << '\n';

  // Assert values
  for (int i = 0; i < entries; ++i) {
    std::cout << i << ": " << ab[i] << '\n';
    //std::cout << i << "my_o_a := " << (float *) my_o_a[i] << '\n';
  }
  for (int i = 0; i < entries; ++i) {
    if (!occa::areBitwiseEqual(ab[i], a[i] + b[i])) {
      throw 1;
    }
  }

  // Free host memory
  delete [] a;
  delete [] b;
  delete [] ab;

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

// Function to convert hexadecimal to decimal

long hexadecimalToDecimal(std::string hexVal)
{
    int len = hexVal.size();

    // Initializing base value to 1, i.e 16^0
    int base = 1;

    long dec_val = 0;

    // Extracting characters as digits from last
    // character
    for (int i = len - 1; i >= 0; i--) {
        // if character lies in '0'-'9', converting
        // it to integral 0-9 by subtracting 48 from
        // ASCII value
        if (hexVal[i] >= '0' && hexVal[i] <= '9') {
            dec_val += (int(hexVal[i]) - 48) * base;

            // incrementing base by power
            base = base * 16;
        }

        // if character lies in 'A'-'F' , converting
        // it to integral 10 - 15 by subtracting 55
        // from ASCII value
        else if (hexVal[i] >= 'A' && hexVal[i] <= 'F') {
            dec_val += (int(hexVal[i]) - 55) * base;

            // incrementing base by power
            base = base * 16;
        }
    }
    return dec_val;
}
