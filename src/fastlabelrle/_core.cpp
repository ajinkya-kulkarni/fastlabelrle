#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_20_API_VERSION
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Run {
    std::size_t start;
    std::size_t length;
};

template <typename Label>
using RunMap = std::unordered_map<Label, std::vector<Run>>;

std::string coco_compress(const std::vector<std::size_t>& counts) {
    std::string out;
    out.reserve(counts.size() * 2);

    for (std::size_t i = 0; i < counts.size(); ++i) {
        std::int64_t x = static_cast<std::int64_t>(counts[i]);
        if (i > 2) {
            x -= static_cast<std::int64_t>(counts[i - 2]);
        }

        bool more = true;
        while (more) {
            std::uint8_t c = static_cast<std::uint8_t>(x & 0x1f);
            x >>= 5;
            more = (c & 0x10) != 0 ? x != -1 : x != 0;
            if (more) {
                c = static_cast<std::uint8_t>(c | 0x20);
            }
            out.push_back(static_cast<char>(c + 48));
        }
    }
    return out;
}

template <typename Label>
void collect_runs(
    const Label* data,
    std::size_t height,
    std::size_t width,
    RunMap<Label>& runs,
    std::vector<Label>& ids
) {
    const std::size_t total = height * width;
    const std::size_t reserve_hint = std::min<std::size_t>(total / 32 + 1, 262144);
    runs.reserve(reserve_hint);
    ids.reserve(reserve_hint);

    // COCO RLE uses Fortran order: top-to-bottom within each column.
    for (std::size_t x = 0; x < width; ++x) {
        std::size_t y = 0;
        while (y < height) {
            const Label label = data[y * width + x];
            const std::size_t start_y = y;
            ++y;
            while (y < height && data[y * width + x] == label) {
                ++y;
            }

            if (label == Label{0}) {
                continue;
            }

            const std::size_t start = x * height + start_y;
            const std::size_t length = y - start_y;
            auto [it, inserted] = runs.try_emplace(label);
            if (inserted) {
                ids.push_back(label);
            }

            auto& label_runs = it->second;
            if (!label_runs.empty()) {
                Run& previous = label_runs.back();
                if (previous.start + previous.length == start) {
                    previous.length += length;
                    continue;
                }
            }
            label_runs.push_back(Run{start, length});
        }
    }
}

std::vector<std::size_t> build_counts(const std::vector<Run>& runs, std::size_t total) {
    std::vector<std::size_t> counts;
    counts.reserve(runs.size() * 2 + 1);

    std::size_t cursor = 0;
    for (const Run& run : runs) {
        counts.push_back(run.start - cursor);
        counts.push_back(run.length);
        cursor = run.start + run.length;
    }
    if (cursor < total) {
        counts.push_back(total - cursor);
    }
    return counts;
}

template <typename Label>
PyObject* encode_impl(PyArrayObject* arr, int typenum) {
    const auto* data = reinterpret_cast<const Label*>(PyArray_DATA(arr));
    const auto height = static_cast<std::size_t>(PyArray_DIM(arr, 0));
    const auto width = static_cast<std::size_t>(PyArray_DIM(arr, 1));

    if (height != 0 && width > std::numeric_limits<std::size_t>::max() / height) {
        PyErr_SetString(PyExc_OverflowError, "labels shape is too large");
        return nullptr;
    }
    const std::size_t total = height * width;

    RunMap<Label> runs;
    std::vector<Label> ids;
    std::vector<std::string> encoded;

    enum class CppError { none, bad_alloc, overflow, unexpected };
    CppError error = CppError::none;

    Py_BEGIN_ALLOW_THREADS
    try {
        collect_runs(data, height, width, runs, ids);
        std::sort(ids.begin(), ids.end());
        encoded.reserve(ids.size());
        for (const Label id : ids) {
            const auto counts = build_counts(runs.at(id), total);
            for (const std::size_t count : counts) {
                if (count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
                    error = CppError::overflow;
                    break;
                }
            }
            if (error != CppError::none) {
                break;
            }
            encoded.push_back(coco_compress(counts));
        }
    } catch (const std::bad_alloc&) {
        error = CppError::bad_alloc;
    } catch (...) {
        error = CppError::unexpected;
    }
    Py_END_ALLOW_THREADS

    if (error == CppError::bad_alloc) {
        return PyErr_NoMemory();
    }
    if (error == CppError::overflow) {
        PyErr_SetString(PyExc_OverflowError, "RLE count exceeds supported range");
        return nullptr;
    }
    if (error == CppError::unexpected) {
        PyErr_SetString(PyExc_RuntimeError, "unexpected C++ error in RLE encoder");
        return nullptr;
    }

    npy_intp dims[1] = {static_cast<npy_intp>(ids.size())};
    PyObject* ids_obj = PyArray_SimpleNew(1, dims, typenum);
    if (ids_obj == nullptr) {
        return nullptr;
    }
    auto* out_ids = reinterpret_cast<Label*>(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(ids_obj))
    );
    std::copy(ids.begin(), ids.end(), out_ids);

    PyObject* counts_obj = PyList_New(static_cast<Py_ssize_t>(encoded.size()));
    if (counts_obj == nullptr) {
        Py_DECREF(ids_obj);
        return nullptr;
    }
    for (std::size_t i = 0; i < encoded.size(); ++i) {
        const std::string& value = encoded[i];
        PyObject* item = PyBytes_FromStringAndSize(
            value.data(), static_cast<Py_ssize_t>(value.size())
        );
        if (item == nullptr) {
            Py_DECREF(ids_obj);
            Py_DECREF(counts_obj);
            return nullptr;
        }
        PyList_SET_ITEM(counts_obj, static_cast<Py_ssize_t>(i), item);
    }

    PyObject* result = PyTuple_New(2);
    if (result == nullptr) {
        Py_DECREF(ids_obj);
        Py_DECREF(counts_obj);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, ids_obj);
    PyTuple_SET_ITEM(result, 1, counts_obj);
    return result;
}

PyObject* py_encode(PyObject*, PyObject* args) {
    PyObject* obj = nullptr;
    if (!PyArg_ParseTuple(args, "O:encode", &obj)) {
        return nullptr;
    }
    if (!PyArray_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "labels must be a NumPy array");
        return nullptr;
    }

    auto* arr = reinterpret_cast<PyArrayObject*>(obj);
    if (PyArray_NDIM(arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "labels must be a 2D array");
        return nullptr;
    }
    if (!PyArray_ISCARRAY_RO(arr)) {
        PyErr_SetString(PyExc_ValueError, "labels must be C-contiguous and aligned");
        return nullptr;
    }
    if (!PyArray_ISNBO(PyArray_DESCR(arr)->byteorder)) {
        PyErr_SetString(PyExc_ValueError, "labels must have native byte order");
        return nullptr;
    }

    const int typenum = PyArray_TYPE(arr);
    if (typenum == NPY_UINT32) {
        return encode_impl<std::uint32_t>(arr, typenum);
    }
    if (typenum == NPY_UINT64) {
        return encode_impl<std::uint64_t>(arr, typenum);
    }

    PyErr_SetString(PyExc_TypeError, "labels dtype must be uint32 or uint64");
    return nullptr;
}

PyMethodDef methods[] = {
    {
        "encode",
        py_encode,
        METH_VARARGS,
        "Encode all nonzero integer labels directly to compressed COCO RLE counts."
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "fastlabelrle C++ core",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__core(void) {
    import_array();
    return PyModule_Create(&module);
}
