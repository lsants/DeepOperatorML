import os
import platform
from pathlib import Path
from ctypes import CDLL, c_double, c_long, POINTER, byref

def influence(c11_val, c12_val, c13_val, c33_val, c44_val,
               dens_val, damp_val,
               r_campo_val, z_campo_val,
               z_fonte_val, r_fonte_val, l_fonte_val,
               freq_val,
               bvptype_val, loadtype_val, component_val):
    
    current_dir = Path(__file__).parent
    system = platform.system()

    if system == 'Windows':
        lib_name = 'axsgrsce.dll'
    elif system == 'Darwin':
        lib_name = 'axsgrsce.dylib'
    elif system == 'Linux':
        lib_name = 'axsgrsce.so'
    else:
        raise OSError('Unsupported operating system')
    
    lib_path = current_dir / 'libs' / lib_name
    
    if not lib_path.exists():
        raise FileNotFoundError(f"Library {lib_name} not found at {lib_path}")

    lib = CDLL(lib_path)

    lib.axsanisgreen.argtypes = [
        POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), #c11,c12,c13,c33,c44
        POINTER(c_double), POINTER(c_double), #dens, damp
        POINTER(c_double), POINTER(c_double), #r, z
        POINTER(c_double), POINTER(c_double), POINTER(c_double),  # h, loadr, loadh
        POINTER(c_double), #omega
        POINTER(c_long), POINTER(c_long), POINTER(c_long), #bvptype_val, loadtype_val, component_val
        POINTER(c_double), POINTER(c_double) # outputs: resultr and resulti
    ]
    lib.axsanisgreen.restype = None

    c11 = c_double(c11_val)
    c12 = c_double(c12_val)
    c13 = c_double(c13_val)
    c33 = c_double(c33_val)
    c44 = c_double(c44_val)
    dens = c_double(dens_val)
    damp = c_double(damp_val)
    r = c_double(r_campo_val)
    z = c_double(z_campo_val)
    h = c_double(z_fonte_val)
    loadr = c_double(r_fonte_val)
    loadh = c_double(l_fonte_val)
    omega = c_double(freq_val)
    bvptype = c_long(bvptype_val)
    loadtype = c_long(loadtype_val)
    component = c_long(component_val)

    resultr = c_double()
    resulti = c_double()


    lib.axsanisgreen(
        byref(c11), byref(c12), byref(c13), byref(c33), byref(c44),
        byref(dens), byref(damp),
        byref(r), byref(z),
        byref(h), byref(loadr), byref(loadh),
        byref(omega),
        byref(bvptype), byref(loadtype), byref(component),
        byref(resultr), byref(resulti)
    )

    wd = resultr.value + 1j * resulti.value
    return wd