# Changes to mm3 code:

## 1. Deprecated TIFF package
### Error:
    File "./aux/mm3_nd2ToTIFF.py", line 20, in <module> from skimage.external import tifffile as tiff ModuleNotFoundError: No module name 'skimage.external'

### Resolution:
    conda install -c conda-forge tifffile
    Remove “from skimage.external”
    The current TIFF package: https://pypi.org/project/tifffile/

## 2. Multi-processing not working
### Resolution:
    Disabled multi-processing in Compile.py, ChannelPicker.py, segment-otsu.py and track-standard.py

## 3. New TIFF package
### Description:
    Because of the updated TIFF package, we cannot access metadata using tif[0].image_decription.

### Resolution:
    Now, we express TIFs as pages and tags and read a particular tag as following:
```
    for tag in tif.pages[0].tags:
        if tag.name=="ImageDescription":
           idata=tag.value
           break
    idata = json.loads(idata)
    return idata
```


## 4. Key discrepancies in .yaml config file
### Resolution:
    Corrected the ['otsu'] related keys in the params.yaml file.


## 5. Changes to skimage.measure regionprops
### Description:
    Regionprops uses RC rather than xy coordinates since version 0.17.

### Resolution:
    Compute region orientation as pi/2 - previous orientation in feretdiameter()
