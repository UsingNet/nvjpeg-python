#include <stdio.h>
#include <malloc.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <Python.h>
#include <pythread.h>
#include <structmember.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <JpegCoder.hpp>

typedef struct
{
    PyObject_HEAD
    JpegCoder* m_handle;
}NvJpeg;


static PyMemberDef NvJpeg_DataMembers[] =
{
        {(char*)"m_handle",   T_OBJECT, offsetof(NvJpeg, m_handle),   0, (char*)"NvJpeg handle ptr"},
        {NULL, 0, 0, 0, NULL}
};

int NvJpeg_init(PyObject *self, PyObject *args, PyObject *kwds) {
  ((NvJpeg*)self)->m_handle = new JpegCoder();
  return 0;
}


static void NvJpeg_Destruct(PyObject* self)
{
    delete (JpegCoder*)(((NvJpeg*)self)->m_handle);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* NvJpeg_Str(PyObject* Self)
{
    return Py_BuildValue("s", "<nvjpeg-python.nvjpeg>");
}

static PyObject* NvJpeg_Repr(PyObject* Self)
{
    return NvJpeg_Str(Self);
}

static PyObject* NvJpeg_decode(NvJpeg* Self, PyObject* Argvs)
{
    JpegCoder* m_handle = (JpegCoder*)Self->m_handle;
    
    unsigned char* jpegData;
    int len;
    if(!PyArg_ParseTuple(Argvs, "y#", &jpegData, &len)){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should jpegData byte string!");
        return NULL;
    }
    JpegCoderImage* img;
    try{
        m_handle->ensureThread(PyThread_get_thread_ident());
        img = m_handle->decode((const unsigned char*)jpegData, len);
    }catch(JpegCoderError e){
        PyErr_Format(PyExc_ValueError, "%s, Code: %d", e.what(), e.code());
        return NULL;
    }

    unsigned char* data = img->buffer();

    npy_intp dims[3] = {(npy_intp)(img->height), (npy_intp)(img->width), 3};
    PyObject* temp = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, data);

    PyArray_ENABLEFLAGS((PyArrayObject*) temp, NPY_ARRAY_OWNDATA);
    delete(img);
    return temp;
}

static PyObject* NvJpeg_encode(NvJpeg* Self, PyObject* Argvs)
{
    PyArrayObject *vecin;
    unsigned int quality = 70;
    if (!PyArg_ParseTuple(Argvs, "O!|I", &PyArray_Type, &vecin, &quality)){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should pass BGR image numpy array!");
        return NULL;
    }

    if (NULL == vecin){
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (PyArray_NDIM(vecin) != 3){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should pass BGR image numpy array by height*width*channel!");
        return NULL;
    }

    if(quality>100){
        quality = 100;
    }

    JpegCoder* m_handle = (JpegCoder*)Self->m_handle;

    PyObject* bytes = PyObject_CallMethod((PyObject*)vecin, "tobytes", NULL);

    int length;
    unsigned char* buffer;
    PyArg_Parse(bytes, "y#", &buffer, &length);    
    auto img = new JpegCoderImage(PyArray_DIM(vecin, 1), PyArray_DIM(vecin, 0), 3, JPEGCODER_CSS_444);
    img->fill(buffer);
    Py_DECREF(bytes);

    m_handle->ensureThread(PyThread_get_thread_ident());
    auto data = m_handle->encode(img, quality);

    PyObject* rtn = PyBytes_FromStringAndSize((const char*)data->data, data->size);

    delete(data);
    delete(img);
    
    return rtn;
}

static PyObject* NvJpeg_read(NvJpeg* Self, PyObject* Argvs)
{
    JpegCoder* m_handle = (JpegCoder*)Self->m_handle;
    
    unsigned char* jpegFile;
    if(!PyArg_ParseTuple(Argvs, "s", &jpegFile)){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should pass jpeg file path string!");
        return NULL;
    }

    FILE* fp = fopen((const char*)jpegFile, "rb");
    if (fp == NULL){
        PyErr_Format(PyExc_IOError, "Cannot open file \"%s\"", jpegFile);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    size_t dataLength = ftell(fp);
    unsigned char* jpegData = (unsigned char*)malloc(dataLength);
    if(jpegData == NULL){
        fclose(fp);
        PyErr_Format(PyExc_IOError, "Out of memeroy when read file \"%s\"", jpegFile);
        return NULL;
    }

    fseek(fp, 0, SEEK_SET);
    if(fread(jpegData, 1, dataLength, fp) != dataLength){
        fclose(fp);
        free(jpegData);
        PyErr_Format(PyExc_IOError, "Read file \"%s\" with error", jpegFile);
        return NULL;
    }

    fclose(fp);

    m_handle->ensureThread(PyThread_get_thread_ident());
    auto img = m_handle->decode((const unsigned char*)jpegData, dataLength);

    free(jpegData);

    unsigned char* data = img->buffer();

    npy_intp dims[3] = {(npy_intp)(img->height), (npy_intp)(img->width), 3};
    PyObject* temp = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, data);

    PyArray_ENABLEFLAGS((PyArrayObject*) temp, NPY_ARRAY_OWNDATA);
    delete(img);
    return temp;
}

static PyObject* NvJpeg_write(NvJpeg* Self, PyObject* Argvs)
{
    unsigned char* jpegFile;
    PyArrayObject *vecin;
    unsigned int quality = 70;
    if (!PyArg_ParseTuple(Argvs, "sO!|I", &jpegFile, &PyArray_Type, &vecin, &quality)){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should pass BGR image numpy array!");
        return NULL;
    }

    FILE* fp = fopen((const char*)jpegFile, "wb");
    if(fp == NULL){
        PyErr_Format(PyExc_IOError, "Cannot open file \"%s\"", jpegFile);
        return NULL;
    }
    
    PyObject* passAvgs = PyTuple_GetSlice(Argvs, 1, 2);
    PyObject* encodeResponse = NvJpeg_encode(Self, passAvgs);
    Py_DECREF(passAvgs);
    if(encodeResponse == NULL){
        fclose(fp);
        return NULL;
    }

    char* jpegData;
    Py_ssize_t jpegDataSize;
    PyBytes_AsStringAndSize(encodeResponse, &jpegData, &jpegDataSize);
    ssize_t write_size = fwrite(jpegData, 1, jpegDataSize, fp);
    if(write_size != jpegDataSize){
        PyErr_Format(PyExc_IOError, "Write file \"%s\" with error", jpegFile);
    }
    Py_DECREF(encodeResponse);
    fclose(fp);
    return Py_BuildValue("l", (long)jpegDataSize);
}


static PyMethodDef NvJpeg_MethodMembers[] =
{
        {"encode",  (PyCFunction)NvJpeg_encode,  METH_VARARGS,  "encode jpge"},
        {"decode", (PyCFunction)NvJpeg_decode, METH_VARARGS,  "decode jpeg"},
        {"read", (PyCFunction)NvJpeg_read, METH_VARARGS,  "read jpeg file and decode"},
        {"write", (PyCFunction)NvJpeg_write, METH_VARARGS,  "encode and write jpeg file"},
        {NULL, NULL, 0, NULL}
};


static PyTypeObject NvJpeg_ClassInfo =
{
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name      = "nvjpeg.NvJpeg",
        .tp_basicsize = sizeof(NvJpeg),
        .tp_itemsize = 0,
        .tp_dealloc   = NvJpeg_Destruct,
        .tp_print   = NULL,
        .tp_getattr = NULL,
        .tp_setattr = NULL,
        .tp_as_async = NULL,
        .tp_repr      = NvJpeg_Repr,
        .tp_as_number = NULL,
        .tp_as_sequence = NULL,
        .tp_as_mapping = NULL,
        .tp_hash = NULL,
        .tp_call = NULL,
        .tp_str       = NvJpeg_Str,
        .tp_getattro = NULL,
        .tp_setattro = NULL,
        .tp_as_buffer = NULL,

        .tp_flags     = Py_TPFLAGS_DEFAULT,
        .tp_doc       = "NvJpeg Python Objects---Extensioned by nvjpeg",
        .tp_traverse  = NULL,
        .tp_clear = NULL,
        .tp_richcompare = NULL,
        .tp_weaklistoffset = 0,
        .tp_iter = NULL,
        .tp_iternext = NULL,

        .tp_methods   = NvJpeg_MethodMembers,
        .tp_members   = NvJpeg_DataMembers,
        .tp_getset = NULL,
        .tp_base = NULL,
        .tp_dict = NULL,
        .tp_descr_get = NULL,
        .tp_descr_set = NULL,
        .tp_dictoffset = 0,

        .tp_init      = NvJpeg_init,
        .tp_alloc   = NULL,
        .tp_new = PyType_GenericNew,
        .tp_free = NULL,
        .tp_is_gc = NULL,
        .tp_bases = NULL,
        .tp_mro = NULL,
        .tp_cache = NULL,
        .tp_subclasses = NULL,
        .tp_weaklist = NULL,
        .tp_del = NULL
};


void NvJpeg_module_destroy(void *_){
    JpegCoder::cleanUpEnv();
}

static PyModuleDef ModuleInfo =
{
        PyModuleDef_HEAD_INIT,
        "NvJpeg Module",
        "NvJpeg by Nvjpeg",
        -1,
        NULL, NULL, NULL, NULL,
        NvJpeg_module_destroy
};

PyMODINIT_FUNC
PyInit_nvjpeg(void) {
    PyObject * pReturn = NULL;

    if(PyType_Ready(&NvJpeg_ClassInfo) < 0) 
        return NULL;

    pReturn = PyModule_Create(&ModuleInfo);
    if(pReturn == NULL)
        return NULL;

    Py_INCREF(&ModuleInfo);

    Py_INCREF(&NvJpeg_ClassInfo);
    if (PyModule_AddObject(pReturn, "NvJpeg", (PyObject*)&NvJpeg_ClassInfo) < 0) {
        Py_DECREF(&NvJpeg_ClassInfo);
        Py_DECREF(pReturn);
        return NULL;
    }

    import_array();
    return pReturn;
}
