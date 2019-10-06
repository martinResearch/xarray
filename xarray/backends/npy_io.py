

import numpy as np
import xarray as xr
import pandas as pd
import sys
import json
import os
import datetime

from xarray.core.utils import (
    decode_numpy_dict_values,
    either_dict_or_kwargs,
    ensure_us_time_resolution,
)
from numpy.compat import (
    asbytes, asstr, asunicode, bytes, basestring, os_fspath, os_PathLike,
    pickle, contextlib_nullcontext
    )
from numpy.lib import format

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime.datetime):
            return obj.__str__()     
        if isinstance(obj, np.datetime64):
            return obj.__str__()         
        return json.JSONEncoder.default(self, obj)


def _is_string_like(obj):
    """
    Check whether obj behaves like a string.
    """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True

def myJsonConverter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()



def save_npys(file, data, compress=False,min_dims_coord_npy = 2):
    if isinstance(data,xr.DataArray):
        _save_dataarray(file, data, compress=compress,min_dims_coord_npy=min_dims_coord_npy)
    elif isinstance(data,xr.Dataset):
        _save_dataset(file, data, compress=compress,min_dims_coord_npy=min_dims_coord_npy)
    else:
        raise BaseException('Unexpected type %'%str(type(data)))
    
class zip_file():
    def __init__(self,file, *args, **kwargs):
        """
        Create a ZipFile.
    
        Allows for Zip64, and the `file` argument can accept file, str, or
        pathlib.Path objects. `args` and `kwargs` are passed to the zipfile.ZipFile
        constructor.
        """        
        if not hasattr(file, 'read'):
            file = os_fspath(file)
        import zipfile
        kwargs['allowZip64'] = True
        file_dir, file_prefix = os.path.split(file) if _is_string_like(file) else (None, 'tmp')
        self.file_dir = file_dir
        self.file_prefix = file_prefix
        self.zipf = zipfile.ZipFile(file, *args, **kwargs)  

    def close(self):
        self.zipf.close()
    def open(self,x):
        return self.zipf.open(x)
    def read(self,x):
        return self.zipf.read(x)
    def namelist(self):
        return self.zipf.namelist()
    def add_bin_data(self,fname,data_bytes):
        if sys.version_info >= (3, 6):
            with self.zipf.open(fname, 'w', force_zip64=True) as fid:
                fid.write(data_bytes)
        else:
            import tempfile
            fd, tmpfile = tempfile.mkstemp(prefix=self.file_prefix, dir=self.file_dir, suffix=fname)
            os.close(fd)             
            try:               
                fid = open(tmpfile, 'wb')                
                try:                
                    fid.write(data_bytes) 
                    fid.close()
                    fid = None
                    self.zipf.write(tmpfile, arcname=fname)
                except IOError as exc:
                    raise IOError("Failed to write to %s: %s" % (tmpfile, exc))
                finally:
                    if fid:
                        fid.close()    
            finally:
                os.remove(tmpfile) 
                
    def add_npy(self,fname,val):
        if sys.version_info >= (3, 6):
            with self.zipf.open(fname, 'w', force_zip64=True) as fid:
                    format.write_array(fid, np.asanyarray(val), allow_pickle=False, pickle_kwargs=None)
        else:       
            import tempfile
            # Since target file might be big enough to exceed capacity of a global
            # temporary directory, create temp file side-by-side with the target file.
            fd, tmpfile = tempfile.mkstemp(prefix=self.file_prefix, dir=self.file_dir, suffix=fname)
            os.close(fd)        
            try:                
                fid = open(tmpfile, 'wb')
                try:
                    format.write_array(fid, np.asanyarray(val), allow_pickle=False, pickle_kwargs=None)
                    fid.close()
                    fid = None
                    self.zipf.write(tmpfile, arcname=fname)
                except IOError as exc:
                    raise IOError("Failed to write to %s: %s" % (tmpfile, exc))
                finally:
                    if fid:
                        fid.close()
            finally:
                os.remove(tmpfile)        
        
def _save_dataarray(file, dataarray, compress=False, min_dims_coord_npy =2):#mostly copied from _savez in numpy\lib\npyio.py
    import zipfile

    if not hasattr(file, 'write'):
        file = os_fspath(file)
        if not file.endswith('.xar'):
            file = file + '.xar'
    if compress:
        compression = zipfile.ZIP_DEFLATED
    else:
        compression = zipfile.ZIP_STORED

    zipf = zip_file(file, mode="w", compression=compression)
    if dataarray.name is None:
        data_name = 'data'             
    else:
        data_name = dataarray.name   
    zipf.add_npy(data_name+'.npy',dataarray.values)
        
    d = dataarray.variable.to_dict(data=False)
    d['version'] = xr.__version__
    d.update({"coords": {}, "name": dataarray.name})
    for k in dataarray.coords:
        assert(k!=data_name)
        coord_var = dataarray.coords[k].variable        
        item = {"attrs": decode_numpy_dict_values(coord_var.attrs), "dtype":str(coord_var.values.dtype)}# we save the type here 
        if (coord_var.dims!=()) and( len(coord_var.dims)>1 or coord_var.dims[0]!=k): # we don't keep the dims if we have a dimension_coordinate or if dims is empty to keep the json more concise (see http://xarray.pydata.org/en/stable/data-structures.html#coordinates)
            item['dims'] = coord_var.dims
        if (coord_var.dims!=()) and len(coord_var.dims)>=min_dims_coord_npy:
            zipf.add_npy(k+'.npy',coord_var.values)
        else:    
            item["data"] = ensure_us_time_resolution(coord_var.values)   # keeping coordinates data in the json      
        d["coords"][k] = item         
    
    json_str = json.dumps(d,cls=NumpyEncoder) + "\n"               # 2. string (i.e. JSON)
    json_bytes = json_str.encode('utf-8') 
    zipf.add_bin_data('DataArray.json',json_bytes)
    zipf.close()

def _save_dataset(file, dataset, compress=False, min_dims_coord_npy = 2):#mostly copied from _savez in numpy\lib\npyio.py
    import zipfile

    if not hasattr(file, 'write'):
        file = os_fspath(file)
        if not file.endswith('.xar'):
            file = file + '.xar'

    if compress:
        compression = zipfile.ZIP_DEFLATED
    else:
        compression = zipfile.ZIP_STORED

    zipf = zip_file(file, mode="w", compression=compression)    
    dataset_dict = dataset.to_dict(data = False)
    dataset_dict['version'] = xr.__version__
    for key, array in dict(dataset.data_vars).items():
            val = np.asanyarray(array.values)  
            if val.ndim >= min_dims_coord_npy:
                zipf.add_npy('%s.npy'%key, val)
            else:
                dataset_dict['data_vars'][key]['data']=ensure_us_time_resolution(val) 
    for key, array in dict(dataset.coords).items():
        val = np.asanyarray(array.values)  
        if val.ndim >= min_dims_coord_npy:
            zipf.add_npy('%s.npy'%key, val)
        else:
            dataset_dict['coords'][key]['data']=ensure_us_time_resolution(val)  
    json_str = json.dumps(dataset_dict,cls=NumpyEncoder) + "\n"              
    json_bytes = json_str.encode('utf-8') 
    zipf.add_bin_data('Dataset.json', json_bytes)
    zipf.close()     


def load_npys(file):     
    # TODO: Use contextlib.ExitStack once we drop Python 2
    if hasattr(file, 'read'):
        fid = file
        own_fid = False
    else:
        fid = open(os_fspath(file), "rb")
        own_fid = True

    if True:
        # Code to distinguish from NumPy binary files and pickles.
        _ZIP_PREFIX = b'PK\x03\x04'
        _ZIP_SUFFIX = b'PK\x05\x06' # empty zip files start with this
        N = len(format.MAGIC_PREFIX)
        magic = fid.read(N)
        # If the file size is less than N, we need to make sure not
        # to seek past the beginning of the file
        fid.seek(-min(N, len(magic)), 1)  # back-up
        if magic.startswith(_ZIP_PREFIX) or magic.startswith(_ZIP_SUFFIX):
            _zip = zip_file(fid)
            files = _zip.namelist()
            _data_dict={}      
            _type = None
            for x in files:
                if x.endswith('.npy'):                    
                    bytes = _zip.open(x)
                    magic = bytes.read(len(format.MAGIC_PREFIX))
                    bytes.close()
                    assert( magic == format.MAGIC_PREFIX)
                    bytes = _zip.open(x)
                    _data_dict[x[:-4]] = format.read_array(bytes, allow_pickle=False, pickle_kwargs=None)                 
                elif x=='Dataset.json':
                    assert(_type is None)
                    _type = xr.Dataset
                    header = json.loads(_zip.read(x))                       
                elif x=='DataArray.json':
                    assert(_type is None)
                    _type = xr.DataArray
                    header = json.loads(_zip.read(x))
            if _type is None:
                raise IOError("Failed to read file")
            if _type ==  xr.DataArray:
                if 'name' in header and (header['name'] is not None):
                    data_name = header['name']
                else:
                    data_name = 'data'                     
                data = _data_dict[data_name]
                assert (data.dtype==header['dtype'])
                assert (data.shape==tuple(header['shape']))
                coords={}
                for k,coord in header['coords'].items():
                    if 'data' in coord:
                        coord_data = np.array(coord['data'],dtype=coord['dtype'])
                    else:
                        coord_data = _data_dict[k]
                    if 'dims' in coord:
                        dims=coord['dims']
                    elif coord_data.ndim==0:
                        dims=()
                    else:
                        dims= [k]
                    coords[k]=xr.DataArray(coord_data,dims=dims) 
                return xr.DataArray(data, coords = coords, dims=header['dims'],attrs=header['attrs'],name=header['name'])
            else: # type is Dataset
                coords={}
                data_vars={}
                for k,d in header['coords'].items():
                    if 'data' in d:
                        data = np.array(d['data'],dtype=d['dtype'])
                    else:
                        data = _data_dict[k]
                    coords[k]=xr.DataArray(data, dims=d['dims'], attrs=d['attrs'])
                for k,d in header['data_vars'].items():
                    if  'data' in d:
                        data = np.array(d['data'],dtype=d['dtype'])
                    else:
                        data = _data_dict[k]
                    data_vars[k]=xr.DataArray(data, dims=d['dims'], attrs=d['attrs'])            
                return xr.Dataset(data_vars, coords=coords,attrs=header['attrs'])
        else:
            raise IOError(
                    "Failed to interpret file %s as a zip" % repr(file))

    return None

def test():
    from xarray.testing import assert_identical
    data = np.random.rand(4, 3)
    
    locs = ['IA', 'IL', 'IN']
    times = pd.date_range('2000-01-01', periods=4)
    foo = xr.DataArray(data, coords=[times, locs], dims=['time', 'space'])
    v=foo.coords['time'].variable
    
    save_npys('foo',foo)
      
    foo_loaded = load_npys('foo.xar')
    assert_identical(foo,foo_loaded)
        
    temp = 15 + 8 * np.random.randn(2, 2, 3)
    lon = [[-99.83, -99.32], [-99.79, -99.23]]
    lat = [[42.25, 42.21], [42.63, 42.59]]
    da = xr.DataArray(temp,name='precipitations',dims=['x','y','time'],
                    coords={'long': (['x', 'y'], lon), 'lat': (['x', 'y'], lat), 'time': pd.date_range('2014-09-06', periods=3), 'reference_time': pd.Timestamp('2014-09-05')})
    save_npys('da',da)
    da_loaded=load_npys('da.xar')
    assert_identical(da,da_loaded)
    
    temp = 15 + 8 * np.random.randn(2, 2, 3)
    precip = 10 * np.random.rand(2, 2, 3)
    lon = [[-99.83, -99.32], [-99.79, -99.23]]
    lat = [[42.25, 42.21], [42.63, 42.59]]
    ds = xr.Dataset({'temperature'  : (['x', 'y', 'time'], temp),
                     'precipitation': (['x', 'y', 'time'], precip)},
                     coords={'long': (['x', 'y'], lon), 'lat': (['x', 'y'], lat), 'time': pd.date_range('2014-09-06', periods=3), 'reference_time': pd.Timestamp('2014-09-05')})
    
    save_npys('ds',ds,min_dims_coord_npy=1)
    ds_loaded= load_npys('ds.xar')
    assert_identical(ds, ds_loaded)
    
    
if __name__ == "__main__":
    test()

    
