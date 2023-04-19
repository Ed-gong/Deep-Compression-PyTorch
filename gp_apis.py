import torch as th
import torch.utils.dlpack
import kernel as gpk

def gp_gspmm(g, X, dim0, dim1, inverse, norm):
    X_dl = th.utils.dlpack.to_dlpack(X)

    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.spmm(g, X_dl, res_dl, inverse, norm)  # do not specify the reduce operation

    return res

def gp_gspmmw(g, X, dim0, dim1, op, inverse):
    X_dl = th.utils.dlpack.to_dlpack(X)
    
    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.gspmmw(g, X_dl, res_dl, op, inverse)

    return res

def gp_gspmmw2d(g, X, dim0, dim1, dim2, op, inverse):
    X_dl = th.utils.dlpack.to_dlpack(X)
    
    # declare the output tensor here
    res = th.zeros(dim0, dim1, dim2)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.spmmw2d(g, X_dl, res_dl, op, inverse)

    return res

def gp_gsddmme(g, X, Y, dim0, dim1, op, inverse):
    X_dl = th.utils.dlpack.to_dlpack(X)
    Y_dl = th.utils.dlpack.to_dlpack(Y)
    
    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.gsddmme(g, X_dl, Y_dl, res_dl, op, inverse)

    return res

def gp_gsddmme2d(g, X, Y, dim0, dim1, op, inverse):
    X_dl = th.utils.dlpack.to_dlpack(X)
    Y_dl = th.utils.dlpack.to_dlpack(Y)
    
    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.gsddmme2d(g, X_dl, Y_dl, res_dl, op, inverse)

    return res

def gp_gsddmm(g, X, Y, dim0, dim1, op, inverse):
    X_dl = th.utils.dlpack.to_dlpack(X)
    Y_dl = th.utils.dlpack.to_dlpack(Y)
    
    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.gsddmm(g, X_dl, Y_dl, res_dl, op, inverse)

    return res

def gp_gsddmm2d(g, X, Y, dim0, dim1, op, inverse):
    X_dl = th.utils.dlpack.to_dlpack(X)
    Y_dl = th.utils.dlpack.to_dlpack(Y)
    
    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.sddmm2d(g, X_dl, Y_dl, res_dl, op, inverse)

    return res

def gp_gspmmw_op(g, X, Y, dim0, dim1, op, inverse):
    X_dl = th.utils.dlpack.to_dlpack(X)
    Y_dl = th.utils.dlpack.to_dlpack(Y)
    
    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.gspmmw_op(g, X_dl, Y_dl, res_dl, op, inverse)

    return res

def gp_gspmmw_op2d(g, X, Y, dim0, dim1, dim2, op, inverse):
    X_dl = th.utils.dlpack.to_dlpack(X)
    Y_dl = th.utils.dlpack.to_dlpack(Y)
    
    # declare the output tensor here
    res = th.zeros(dim0, dim1, dim2)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.spmmw_op2d(g, X_dl, Y_dl, res_dl, op, inverse)

    return res

##################################
def gp_spmmw_model(g, X, Y, Z, dim0, dim1, op, inverse):
    X_dl = th.utils.dlpack.to_dlpack(X)
    Y_dl = th.utils.dlpack.to_dlpack(Y)
    Z_dl = th.utils.dlpack.to_dlpack(Z)
 
    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.gspmmw_model(g, X_dl, Y_dl, Z_dl, res_dl, op, inverse)

    return res

def gp_spmmw_model_without_bias(g, X, Y, dim0, dim1, op, inverse):
    X_dl = th.utils.dlpack.to_dlpack(X)
    Y_dl = th.utils.dlpack.to_dlpack(Y)
    #Z_dl = th.utils.dlpack.to_dlpack(Z)
 
    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.gspmmw_model_without_bias(g, X_dl, Y_dl, res_dl, op, inverse)

    return res


def gp_sddmm_model(g, X, Y, dim0, dim1, op):
    X_dl = th.utils.dlpack.to_dlpack(X)
    Y_dl = th.utils.dlpack.to_dlpack(Y)
    
    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.gsddmme_model(g, X_dl, Y_dl, res_dl, op)

    return res
