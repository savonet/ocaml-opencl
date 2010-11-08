#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/callback.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/misc.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>
#include <caml/custom.h>

#include <CL/cl.h>

#include <assert.h>
#include <stdio.h>

#define Val_platform_id(id) (value)id
#define Platform_id_val(id) (cl_platform_id)id

CAMLprim value caml_opencl_num_platforms(value unit)
{
  CAMLparam0();
  int n;

  assert(clGetPlatformIDs(0, NULL, &n) == CL_SUCCESS);

  CAMLreturn(Val_int(n));
}

CAMLprim value caml_opencl_platform_ids(value unit)
{
  CAMLparam0();
  CAMLlocal1(ans);
  int n;
  int i;
  cl_platform_id *ids;

  assert(clGetPlatformIDs(0, NULL, &n) == CL_SUCCESS);
  ids = malloc(n * sizeof(cl_platform_id));
  assert(clGetPlatformIDs(n, ids, NULL) == CL_SUCCESS);
  ans = caml_alloc_tuple(n);
  for (i = 0; i < n; i++)
    Store_field(ans, i, Val_platform_id(ids[i]));
  free(ids);

  CAMLreturn(ans);
}

static cl_platform_info Platform_info_val(value var)
{
  if (var == hash_variant("Profile"))
    return CL_PLATFORM_PROFILE;
  else if (var == hash_variant("Version"))
    return CL_PLATFORM_VERSION;
  else if (var == hash_variant("Name"))
    return CL_PLATFORM_NAME;
  else if (var == hash_variant("Vendor"))
    return CL_PLATFORM_VENDOR;
  else if (var == hash_variant("Extensions"))
    return CL_PLATFORM_EXTENSIONS;
  else
    assert(0);
}

CAMLprim value caml_opencl_platform_info(value _id, value var)
{
  CAMLparam2(_id, var);
  CAMLlocal1(ans);

  cl_platform_id id = Platform_id_val(_id);
  cl_platform_info info;
  int len = 1024;
  char s[len];

  assert(clGetPlatformInfo(id, Platform_info_val(var), len, s, NULL) == CL_SUCCESS);
  ans = caml_copy_string(s);
  CAMLreturn(ans);
}

static cl_context_properties Context_properties_val(value var)
{
  if (var == hash_variant("Platform"))
    return CL_CONTEXT_PLATFORM;
  else
    assert(0);
}

static cl_device_type Device_type_val(value var)
{
  if (var == hash_variant("CPU"))
    return CL_DEVICE_TYPE_CPU;
  else if (var == hash_variant("GPU"))
    return CL_DEVICE_TYPE_GPU;
  else if (var == hash_variant("Accelerator"))
    return CL_DEVICE_TYPE_ACCELERATOR;
  else if (var == hash_variant("Default"))
    return CL_DEVICE_TYPE_DEFAULT;
  else if (var == hash_variant("All"))
    return CL_DEVICE_TYPE_ALL;
  else
    assert(0);
}

#define Val_context(c) (value)c
#define Context_val(v) (cl_context)v

CAMLprim value caml_opencl_create_context_from_type(value platform, value device_type)
{
  CAMLparam2(platform, device_type);
  cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(Platform_id_val(platform)), 0 };
  cl_context context;
  cl_int err;

  context = clCreateContextFromType(cprops, Device_type_val(device_type), NULL, NULL, &err);
  /* TODO: check err */
  assert(context);

  CAMLreturn(Val_context(context));
}

#define Val_device_id(d) (value)d
#define Device_id_val(v) (cl_device_id)v

CAMLprim value caml_opencl_context_devices(value context)
{
  CAMLparam1(context);
  CAMLlocal1(ans);
  size_t ndev;
  cl_device_id *devs;
  int i;

  assert(clGetContextInfo(Context_val(context), CL_CONTEXT_DEVICES, 0, NULL, &ndev) == CL_SUCCESS);
  devs = malloc(ndev * sizeof(cl_device_id));
  assert(clGetContextInfo(Context_val(context), CL_CONTEXT_DEVICES, ndev, devs, NULL) == CL_SUCCESS);
  ans = caml_alloc_tuple(ndev);
  for (i = 0; i < ndev; i++)
    Store_field(ans, i, Val_device_id(devs[i]));
  free(devs);

  CAMLreturn(ans);
}

#define Val_command_queue(q) (value)q
#define Command_queue_val(v) (cl_command_queue)v

CAMLprim value caml_opencl_create_command_queue(value context, value device)
{
  CAMLparam2(context, device);
  cl_command_queue queue;
  cl_int err;

  /* TODO: properties */
  queue = clCreateCommandQueue(Context_val(context), Device_id_val(device), 0, &err);
  /* TODO: check err */
  assert(queue);

  CAMLreturn(Val_command_queue(queue));
}

#define Val_program(p) (value)p
#define Program_val(v) (cl_program)v

CAMLprim value caml_opencl_create_program_with_source(value context, value source)
{
  CAMLparam2(context, source);
  cl_program prog;
  const char *s[1];
  size_t len;
  cl_int err;

  len = caml_string_length(source);
  s[0] = String_val(source);
  prog = clCreateProgramWithSource(Context_val(context), 1, s, &len, &err);
  /* TODO: check err */
  assert(prog);
}

CAMLprim value caml_opencl_build_program(value prog, value devices, value options)
{
  CAMLparam3(prog, devices, options);
  cl_uint ndevs;
  cl_device_id *devs;
  int i;

  ndevs = Wosize_val(devices);
  printf("devs: %d\n", ndevs);
  devs = malloc(ndevs * sizeof(cl_device_id));
  for (i = 0; i < ndevs; i++)
    devs[i] = Device_id_val(Field(devices, i));

  assert(clBuildProgram(Program_val(prog), ndevs, devs, String_val(options), NULL, NULL) == CL_SUCCESS);

  free(devs);
  CAMLreturn(Val_unit);
}

#define Val_kernel(k) (value)k
#define Kernel_val(v) (cl_kernel)v

CAMLprim value caml_opencl_create_kernel(value prog, value name)
{
  CAMLparam2(prog, name);
  cl_kernel kernel;
  cl_int err;

  kernel = clCreateKernel(Program_val(prog), String_val(name), &err);
  assert(kernel && err == CL_SUCCESS);

  CAMLreturn(Val_kernel(kernel));
}
