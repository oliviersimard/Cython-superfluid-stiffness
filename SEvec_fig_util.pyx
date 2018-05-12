from __future__ import print_function
from types import FunctionType
import numpy as np
import os, re
import time, logging
import warnings
import matplotlib.pyplot as plt
from fnmatch import fnmatch
from functools import wraps
from libc.stdlib cimport atoi
from libc.stdio cimport *
from libc.string cimport *
from cpython cimport bool

#-----------------------------------------------------------------------------
#olivier.simard2@usherbrooke.ca

#File containing the classes and methods to produce figures written in cython

#-----------------------------------------------------------------------------

class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
    expr -- input expression in which the error occurred
    msg  -- explanation of the error
    """
    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg

class ArgumentError(Exception):
    """Exception raised for errors in the input arguments.

    Attributes:
    expr -- mistaken argument entered as input parameter
    msg -- explanation of the error  
    """
    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg

def my_time(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        #cdef float t0
        #cdef float tf
        t0 = time.time()
        result = method(*args, **kwargs)
        tf = time.time() - t0
        print('{} ran in: {} sec'.format(method.__name__, tf))
    return wrapped

def my_log(method):
    """Function-based decorator used to create log.file associated with input function"""
    cdef dict dict_el
    logging.basicConfig(filename='{}.log'.format(method.__name__), level=logging.INFO)
    @wraps(method)
    def wrapped(*args, **kwargs):
        mappingproxy = list(Cdmft_data.__dict__.keys())
        for dict_el in mappingproxy:
            path_tmp = os.getcwd()+"/"+dict_el+".log"
            if os.path.exists(path_tmp):
                print(dict_el+".log created")
        logging.info('Ran with args: {}, and kwargs: {}'.format(args,kwargs))
        return method(*args,**kwargs)
    return wrapped

class MetaClass(type):
    """Class used to wrap methods of class with decorator"""
    def __new__(meta, classname, bases, classDict):
        newClassDict = {}
        print("MetaClass.__new__: ", MetaClass.__new__)
        for attributeName, attribute in classDict.items():
            if isinstance(attribute, FunctionType):
                print("attribute: ", attribute)
                attribute = my_time(my_log(attribute))
            newClassDict[attributeName] = attribute
        return type.__new__(meta, classname, bases, newClassDict)

cpdef parse_charptr_to_py_int(char* s):
    assert s is not NULL
    return atoi(s)

cdef extern from "math.h":
    cpdef double sin(double x)

cdef class Cdmft_data(object):
    """Class with attributes and methods to fetch cdmft data and make plots. Pickle protocol is implemented."""
    __metaclass__ = MetaClass #Wrap all the methods
    cdef readonly str paths
    cdef readonly str pattern
    cdef readonly int verbose
    cdef public int w_grid
    cdef public double U
    cdef public int beta
    cdef public float dop_min
    cdef public float dop_max

    def __init__(self, str paths, str pattern, int w_grid, double U, int beta, float dop_min, float dop_max, int verbose):
        """Receives the state of object and applies it to the instance."""
        self.paths = paths; self.pattern = pattern; self.w_grid = w_grid; self.U = U
        self.beta = beta; self.dop_min = dop_min; self.dop_max = dop_max; self.verbose = verbose

    def gen_file_path(self, str key_folder, int verbose):
        """Method that searches the path of the files containing cdmft data."""
        if verbose > 0:
            print("In gen_file_path method\n")
        file_path_list = []
        cdef str path, name
        cdef list subdirs, files
        for path, subdirs, files in os.walk(self.paths):
            for name in files:
                if key_folder in path:
                    if fnmatch(name, self.pattern):
                        file_path_list.append(os.path.join(path,name))
        return file_path_list

    
    def gen_cdmft_data(self, int tr_index_range, str paths, str opt, int verbose):
        """Method that produces generators of the trace of electronic SE or G with respect to doping (HF) or frequency"""
        if verbose > 0:
            print("In gen_cdmft_data method\n")
            print("trace of indices: ", tr_index_range)
        #opt = new char[index_range]
        cdef int i
        cdmft_data = np.load(paths)
        if opt == "freq":
            if verbose > 0:
                print("freq chosen")
            for i in xrange(0,cdmft_data.shape[1]):
                im_el = np.imag(np.trace(cdmft_data[0,i,4:6,4:6])).tolist()
                re_el = np.real(np.trace(cdmft_data[0,i,4:6,4:6])).tolist()
                yield re_el, im_el
        elif opt == "HF":
            if verbose > 0:
                print("HF chosen")
            im_el = np.imag(np.trace(cdmft_data[0,-1,4:6,4:6])).tolist()
            re_el = np.real(np.trace(cdmft_data[0,-1,4:6,4:6])).tolist()
            yield re_el, im_el

        else:
            print("Error in argument")
            raise ArgumentError(opt,"Wrong opt argument input entered. Only \'HF\' and \'freq\' implemented.")

    def same_repository(self, str key_folder, int verbose):
        """Method called when dealing with data stored in same repository."""
        if verbose > 0:
            print("In same_repository method\n")
        regex_not_sparse = re.compile(r'(.*?)(?=/)')
        regex_int = re.compile(r'\d+')
        file_path_list = self.gen_file_path(key_folder, verbose)
        #print("file_path_list", file_path_list)
        int_list = []
        warnings.warn("Files are stored in same repository.")
        if "SEvec" in regex_not_sparse.findall(file_path_list[0]):
            SEvsG = "SEvec"
        elif "Gvec" in regex_not_sparse.findall(file_path_list[0]):
            SEvsG = "Gvec"
        else:
            raise InputError("filestr","Error in input file: check your file path!")
        #One has to sort the files according to their pending number
        for filestr in file_path_list:
            #print("filestr: ", filestr)
            int_list.append(int(regex_int.findall(filestr)[-1]))

        list_file_fusion = [el for el in zip(int_list,file_path_list)]
        list_file_fusion.sort()
        #print("list_file_fusion: ", list_file_fusion)
        return SEvsG, list_file_fusion


    def sparse_repository(self, str key_folder, int verbose):
        """Method used to separate different set of data for later handling."""
        if verbose > 0:
            print("In sparse_repository method\n")
            warnings.warn("Files are stored in different repositories.")
        regex_sparse = re.compile(r'(.*?)(?=_)')
        regex_int = re.compile(r'\d+')
        file_path_list = self.gen_file_path(key_folder, verbose)
        #print("file_path_list: ", file_path_list)
        file_path_list_2 = []
        int_list = []
        int_list_2 = []
        cdef str filestr
        for filestr in file_path_list:
            break_down = regex_sparse.findall(filestr)
            #print("break_down : ", break_down)
            if "s/SEvec" in break_down:
                SEvsG = "SEvec"
                file_path_list_2.append(filestr)
                #file_path_list.pop(file_path_list.index(filestr))
            elif "s/Gvec" in break_down:
                SEvsG = "Gvec"
                file_path_list_2.append(filestr)
        #print("file_path_list_2 : ", file_path_list_2)
        for filestr in file_path_list_2:
            file_path_list.remove(filestr)
        #print("file_path_list : ", file_path_list)
        if len(file_path_list_2) == 0:
            raise InputError("file_str","Error in input file: check your file path!")
        int_list_2 = []
        for filestr in file_path_list:
            int_list.append(int(regex_int.findall(filestr)[-1]))
        for filestr in file_path_list_2:
            int_list_2.append(int(regex_int.findall(filestr)[-1]))

        list_file_fusion = [el for el in zip(int_list,file_path_list)]
        list_file_fusion.sort()
        list_file_fusion_2 = [el for el in zip(int_list_2,file_path_list_2)]
        list_file_fusion_2.sort()

        return SEvsG, np.vstack((list_file_fusion, list_file_fusion_2))

    def repository_differentiation(self, bool sparse, str key_folder, int verbose):
        """Method created to distinguished between data in same repository or in sparse repositories."""
        if verbose > 0:
            print("In repository_differentiation method\n")
        if sparse:
            if verbose > 0:
                print("sparse")
            return self.sparse_repository(key_folder, verbose)
        else:
            if verbose > 0:
                print("not sparse")
            return self.same_repository(key_folder, verbose)
    @my_log
    def load_data_from_files(self, bool sparse, str key_folder, int tr_index_range, str opt, str mu_list, int verbose):
        """Method used to load the data from the files in the targeted range of dopings."""
        if verbose > 0:
            print("In load_data_from_files method\n")
        gen_file_list = []
        dop_range = []
        index_range = []
        cdef int i = 0
        cdef int j
        #cdef str sorted_filestr
        list_mu = np.loadtxt(mu_list,skiprows=0,usecols=(1,))
        #print("list_mu: ", list_mu)
        while i<len(mu_list):
            if list_mu[i]>self.dop_min and list_mu[i]<self.dop_max:
                dop_range.append(list_mu[i])
                index_range.append(i)
            i += 1
        SEvsG, sorted_file_list = self.repository_differentiation(sparse, key_folder, verbose)
        sorted_file_array = np.asarray(sorted_file_list)
        #print("SEvsG, sorted_file_array: ", SEvsG, type(sorted_file_array)) 
        for j, sorted_filestr in enumerate(sorted_file_array[:,1]):
            #print("sorted_filestr: ", sorted_filestr)
            if j in index_range:
                cdmft_data = self.gen_cdmft_data(tr_index_range, str(sorted_filestr), opt, verbose)
                #print("cdmft_data : ", list(cdmft_data))
                gen_file_list.append(cdmft_data)
        dop_gen_file_list = list(sorted(zip(dop_range,gen_file_list),key=lambda x: x[0]))

        return SEvsG, dop_gen_file_list
    @my_time
    def gen_plot(self, bool sparse, str key_folder, int tr_index_range, str opt, str mu_list, str filename, int verbose):
        """Method to plot the RE and IM parts of the SE or G."""
        if verbose > 0:
            print("In gen_plot method \n")
        SEvsG, data_gen_list = self.load_data_from_files(sparse, key_folder, tr_index_range, opt, mu_list, verbose)
        cdef float dop
        cdef tuple dop_gen
        cdef tuple dop_re_el
        cdef float w
        list_dop_re_el_list = []
        #data_gen_array = np.asarray(data_gen_list)
        bbox_props = dict(boxstyle="square,pad=0.3", fc="yellow", ec="b", lw=2)
        color_im = iter(plt.cm.rainbow(np.linspace(0,1,200)))
        color_re = iter(plt.cm.rainbow(np.linspace(0,1,200)))
        fig = plt.figure()
        w_list = [float((2*n+1)*np.pi/(self.beta)) for n in range(self.w_grid)]
        #---------------------------Imaginary part plotted--------------------------
        if opt == "freq":
            axIM = plt.subplot(211)
            if SEvsG == "SEvec":
                ylabel = (r"Im$\Sigma_{%02d}$"r"$\left(\omega\right)$" % (tr_index_range)) # <<<<--------------------------------------------------Modify to fit
                plt.title(r"Self-energy $\omega$-dependence at different dopings for $\beta = {0:2.2f}$".format(self.beta))
            elif SEvsG == "Gvec":
                ylabel = (r"ImG$_{%02d}$"r"$\left(\omega\right)$" % (tr_index_range))
                plt.title(r"Green's function $\omega$-dependence at different dopings for $\beta = {0:2.2f}$".format(self.beta))
            plt.ylabel(ylabel, fontsize=15)
            x_lim = np.ceil((2*self.w_grid + 1)*np.pi/self.beta)
            #print("x_lim : ", x_lim)
            plt.xlim([-0.01,x_lim])
            plt.xticks(np.arange(0,int(x_lim+1),1))
            for dop_gen in data_gen_list:
                im_el_list = []
                re_el_list = []
                dop, gen = dop_gen
                for w in range(self.w_grid):
                    re_el, im_el = next(gen)
                    im_el_list.append(im_el)
                    re_el_list.append(re_el)
                list_dop_re_el_list.append((dop,re_el_list))
                if verbose > 0:
                    print("len_im_el_list: ", len(im_el_list),"\n")
                    print("len_list_re_el_list: ", len(list_dop_re_el_list),"\n")
                axIM.plot(w_list,im_el_list,linestyle="-",marker='o',markersize=3,linewidth=2,color=next(color_im))
        #-----------------------------Real part plotted------------------------------    
            axRE = plt.subplot(212)
            xlabel = (r"$i\omega$")
            if SEvsG == "SEvec":
                ylabel = (r"Re$\Sigma_{%02d}$"r"$\left(\omega\right)$" % (tr_index_range))
            elif SEvsG == "Gvec":
                ylabel = (r"ReG$_{%02d}$"r"$\left(\omega\right)$" % (tr_index_range))
            plt.xlabel(xlabel, fontsize=15)
            plt.ylabel(ylabel, fontsize=15)
            x_lim = np.ceil((2*self.w_grid + 1)*np.pi/self.beta)
            plt.xlim([-0.01,x_lim])
            plt.xticks(np.arange(0,int(x_lim+1),1))
            box = axRE.get_position()
            axRE.set_position([box.x0, box.y0 + box.height*0.1, box.width, box.height*0.9])
            for dop_re_el in list_dop_re_el_list:
                dop, re_el_l = dop_re_el
                axRE.plot(w_list,re_el_l,linestyle="-",marker='o',markersize=3,linewidth=2,color=next(color_re),label=r"$\delta=%.5f$"%(dop))
            axRE.legend(loc='upper center', bbox_to_anchor=(0.5,-0.15), fancybox=True, shadow=True, ncol=12)

        elif opt == "HF":
            regex = re.compile(r'(.*?)(?=/)')
            filename = filename + "HF"
            axHF = plt.subplot(111)
            ylabel = (r"Re$\Sigma_{%02d}$"r"$\left(\omega\to\infty\right)$" % (tr_index_range))
            xlabel = (r"doping $\delta$")
            plt.ylabel(ylabel, fontsize=15)
            plt.xlabel(xlabel, fontsize=15)
            plt.title(r"Re$\Sigma\left(\omega\to\infty\right)$ at different dopings for $\beta = {0:2.2f}$".format(self.beta))
            dop_list = []
            re_el_list = []
            if verbose > 0:
                print("len data_gen_list: ", len(data_gen_list))
            for dop_gen in data_gen_list:
                dop, gen = dop_gen
                dop_list.append(dop)
                re_el, im_el = next(gen)
                if "COEX" in regex.findall(mu_list):
                    re_el_list.append(re_el/(self.U))
                elif "NOCOEX" in regex.findall(mu_list):
                    re_el_list.append(re_el/(self.U))
            axHF.plot(dop_list,re_el_list,linestyle="-",marker='o',markersize=3,linewidth=2,color='black')

        plt.show()
        fig.savefig(filename + ".pdf", format='pdf')
        return None

