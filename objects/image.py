import warnings
import rasterio
import cupy as cp
import numpy as np
import re
from cupy.cuda import Device, Event
from rasterio.warp import calculate_default_transform, reproject, Resampling

class Image:
    def __init__(self, filepath):
        """Initialize the Image object from a raster file."""
        self.filepath = filepath
        self.dataset = None
        self.data = None
        self.meta = None
        self.crs = None
        self.transform = None
        self.bounds = None
        self.nodata = None
        self.band_descriptions = None

        if filepath:
            self._load_image()

    def _load_image(self):
        """Load the image data and metadata from the file."""
        try:
            self.dataset = rasterio.open(self.filepath)
            self.data = self.dataset.read()
            self.meta = self.dataset.meta
            self.crs = self.dataset.crs
            self.transform = self.dataset.transform
            self.bounds = self.dataset.bounds
            self.nodata = self.dataset.nodata
            self.band_descriptions = self.dataset.descriptions
        except rasterio.errors.RasterioIOError as e:
            raise IOError(f"Failed to open the image file: {e}")

    def reduce(self, reducer):
        """Apply a reduction function and return a new Image object with the reduced data."""
        if self.data is None:
            raise ValueError("No image data loaded")

        reducers = {
            'mean': np.nanmean,
            'max': np.nanmax,
            'min': np.nanmin,
            'sum': np.nansum,
            'median': np.nanmedian
        }

        if reducer not in reducers:
            raise ValueError(f"Unsupported reducer: {reducer}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            reduced_data = reducers[reducer](self.data, axis=0)
            
            if np.all(np.isnan(reduced_data)):
                reduced_data = np.full(reduced_data.shape, self.nodata, dtype=self.data.dtype)

        new_meta = self.meta.copy()
        new_meta.update({
            'count': 1,
            'dtype': reduced_data.dtype
        })
        
        new_image = Image(None)
        new_image.data = np.expand_dims(reduced_data, axis=0)
        new_image.meta = new_meta
        new_image.crs = self.crs
        new_image.transform = self.transform
        new_image.bounds = self.bounds
        new_image.nodata = self.nodata
        new_image.band_descriptions = [f"Reduced_{reducer}"]
        
        return new_image

    def get_band_names(self):
        """Return the names of the image bands if available."""
        if self.band_descriptions is None:
            return [f"Band_{i+1}" for i in range(self.data.shape[0])]
        return self.band_descriptions

    def set_band_names(self, names):
        """Set new names for the image bands."""
        if self.data is None:
            raise ValueError("No image data loaded")
        if len(names) != self.data.shape[0]:
            raise ValueError("Number of names does not match number of bands")
        self.band_descriptions = names
        if self.meta:
            self.meta['descriptions'] = names

    def check_dimensions(self):
        """Check and return the dimensions of the image (bands, height, width)."""
        if self.data is None:
            raise ValueError("No image data loaded")
        return self.data.shape

    def get_data(self):
        """Return the image data as a NumPy array."""
        if self.data is None:
            raise ValueError("No image data loaded")
        return self.data

    def map(self, func):
        """Apply a custom function to each pixel in the image."""
        if self.data is None:
            raise ValueError("No image data loaded")
        return func(self.data)

    def save(self, output_filepath):
        """Save the current image data to a new file."""
        if self.data is None:
            raise ValueError("No image data to save")
        
        with rasterio.open(output_filepath, 'w', **self.meta) as dst:
            dst.write(self.data)
            if self.band_descriptions:
                dst.descriptions = self.band_descriptions

    def get_resolution(self):
        """Returns the resolution of the image in the units of its CRS."""
        if self.transform is None:
            raise ValueError("No image transform available")
        return self.transform.a, -self.transform.e

    def image_reproject(self, crs, scale=None):
        """
        Reproject the image to a new coordinate reference system and optionally rescale.

        Args:
            crs (str): The target coordinate reference system (e.g., 'EPSG:4326').
            scale (float or tuple, optional): The target resolution in units of the target CRS.
                If a single number, it's used for both x and y. If a tuple, it's (x_scale, y_scale).

        Returns:
            Image: A new Image object with the reprojected data.
        """
        if self.data is None or self.crs is None or self.transform is None:
            raise ValueError("Image data, CRS, or transform is missing")

        # Determine the target transform and dimensions
        if scale:
            if isinstance(scale, (int, float)):
                scale = (scale, scale)
            dst_affine, dst_width, dst_height = calculate_default_transform(
                self.crs, crs, self.meta['width'], self.meta['height'], 
                *self.bounds, resolution=scale)
        else:
            dst_affine, dst_width, dst_height = calculate_default_transform(
                self.crs, crs, self.meta['width'], self.meta['height'], *self.bounds)

        # Create the destination raster
        dst_data = np.zeros((self.meta['count'], dst_height, dst_width), dtype=self.meta['dtype'])

        # Perform the reprojection
        for i in range(self.meta['count']):
            reproject(
                source=self.data[i],
                destination=dst_data[i],
                src_transform=self.transform,
                src_crs=self.crs,
                dst_transform=dst_affine,
                dst_crs=crs,
                resampling=Resampling.nearest
            )

        # Create a new Image object with the reprojected data
        new_image = Image(None)
        new_image.data = dst_data
        new_image.crs = crs
        new_image.transform = dst_affine
        new_image.meta = self.meta.copy()
        new_image.meta.update({
            'driver': 'GTiff',
            'height': dst_height,
            'width': dst_width,
            'transform': dst_affine,
            'crs': crs
        })
        new_image.bounds = rasterio.transform.array_bounds(dst_height, dst_width, dst_affine)
        new_image.nodata = self.nodata
        new_image.band_descriptions = self.band_descriptions

        return new_image

    def get_band_names(self):
        """Return the names of the image bands if available."""
        if self.band_descriptions is None:
            return [f"Band_{i+1}" for i in range(self.data.shape[0])]
        return self.band_descriptions

    def set_band_names(self, names):
        """Set new names for the image bands."""
        if self.data is None:
            raise ValueError("No image data loaded")
        if len(names) != self.data.shape[0]:
            raise ValueError("Number of names does not match number of bands")
        self.band_descriptions = names
        if self.meta:
            self.meta['descriptions'] = names

    def get_stats(self):
        """Calculate basic statistics for each band."""
        if self.data is None:
            raise ValueError("No image data loaded")
        stats = []
        for band in range(self.data.shape[0]):
            band_data = self.data[band]
            stats.append({
                'min': np.nanmin(band_data),
                'max': np.nanmax(band_data),
                'mean': np.nanmean(band_data),
                'std': np.nanstd(band_data)
            })
        return stats
    
    def select(self, band_name):
        """Select a specific band by name and return a new Image object."""
        if self.band_descriptions is None or band_name not in self.band_descriptions:
            raise ValueError(f"Band '{band_name}' not found")
        
        band_index = self.band_descriptions.index(band_name)
        new_image = Image(None)
        new_image.data = self.data[band_index:band_index+1]
        new_image.meta = self.meta.copy()
        new_image.meta.update({'count': 1})
        new_image.crs = self.crs
        new_image.transform = self.transform
        new_image.bounds = self.bounds
        new_image.nodata = self.nodata
        new_image.band_descriptions = [band_name]
        
        return new_image

    def rename(self, new_name):
        """Rename the band(s) of the image."""
        if isinstance(new_name, str):
            if self.data.shape[0] != 1:
                raise ValueError("Can only rename to a single name if image has one band")
            self.band_descriptions = [new_name]
        elif isinstance(new_name, list):
            if len(new_name) != self.data.shape[0]:
                raise ValueError("Number of new names must match number of bands")
            self.band_descriptions = new_name
        else:
            raise ValueError("new_name must be a string or a list of strings")
        
        return self

    @staticmethod
    def m_expression(expression, variables):
        """Apply a custom expression to create a new Image."""
        numpy_expression = expression.replace('sqrt', 'np.sqrt')
        
        numpy_vars = {key: value.data.squeeze() for key, value in variables.items()}
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            result = eval(numpy_expression, {'np': np}, numpy_vars)
        
        new_image = Image(None)
        new_image.data = np.expand_dims(result, axis=0)
        new_image.meta = next(iter(variables.values())).meta.copy()
        new_image.meta.update({'count': 1, 'dtype': result.dtype})
        new_image.crs = next(iter(variables.values())).crs
        new_image.transform = next(iter(variables.values())).transform
        new_image.bounds = next(iter(variables.values())).bounds
        new_image.nodata = next(iter(variables.values())).nodata
        new_image.band_descriptions = ['expression_result']
        
        return new_image
    
    def get_sample(self, size=(100, 100), position='center'):
        """
        Return a small sample of the image data.

        Args:
            size (tuple): The size of the sample (height, width). Default is (100, 100).
            position (str or tuple): The position of the sample. Can be 'center', 'random',
                                     or a tuple of (row, col) coordinates. Default is 'center'.

        Returns:
            Image: A new Image object containing the sampled data.
        """
        if self.data is None:
            raise ValueError("No image data available")

        bands, height, width = self.data.shape

        if height < size[0] or width < size[1]:
            raise ValueError("Sample size is larger than the image dimensions")

        if position == 'center':
            start_row = (height - size[0]) // 2
            start_col = (width - size[1]) // 2
        elif position == 'random':
            start_row = np.random.randint(0, height - size[0] + 1)
            start_col = np.random.randint(0, width - size[1] + 1)
        elif isinstance(position, tuple) and len(position) == 2:
            start_row, start_col = position
            if start_row < 0 or start_row + size[0] > height or start_col < 0 or start_col + size[1] > width:
                raise ValueError("Invalid position: sample extends beyond image boundaries")
        else:
            raise ValueError("Invalid position argument")

        sample_data = self.data[:, start_row:start_row+size[0], start_col:start_col+size[1]]

        sample_image = Image(None)
        sample_image.data = sample_data
        sample_image.meta = self.meta.copy()
        sample_image.meta.update({
            'height': size[0],
            'width': size[1],
        })
        sample_image.crs = self.crs
        sample_image.transform = self.transform * rasterio.Affine.translation(start_col, start_row)
        sample_image.bounds = None  # Bounds would need to be recalculated
        sample_image.nodata = self.nodata
        sample_image.band_descriptions = self.band_descriptions

        return sample_image
    
    @staticmethod
    def bool_expression(expression, variables):
        """Apply a custom boolean expression to create a new Image."""
        def parse_condition(condition):
            return condition.replace(' and ', ' & ').replace(' or ', ' | ')

        def parse_expression(expr):
            parts = expr.split('?')
            if len(parts) == 1:
                return parts[0].strip()
            condition = parse_condition(parts[0].strip())
            true_false = parts[1].split(':')
            true_expr = parse_expression(true_false[0].strip())
            false_expr = parse_expression(true_false[1].strip() if len(true_false) > 1 else '0')
            return f"np.where({condition}, {true_expr}, {false_expr})"

        numpy_vars = {key: value.data.squeeze() for key, value in variables.items()}
        numpy_expression = parse_expression(expression)

        with np.errstate(divide='ignore', invalid='ignore'):
            result = eval(numpy_expression, {'np': np}, numpy_vars)

        new_image = Image(None)
        new_image.data = np.expand_dims(result, axis=0)
        new_image.meta = next(iter(variables.values())).meta.copy()
        new_image.meta.update({'count': 1, 'dtype': result.dtype})
        new_image.crs = next(iter(variables.values())).crs
        new_image.transform = next(iter(variables.values())).transform
        new_image.bounds = next(iter(variables.values())).bounds
        new_image.nodata = next(iter(variables.values())).nodata
        new_image.band_descriptions = ['expression_result']
        
        return new_image
    
    @staticmethod
    def expression(expression, variables):
        """Apply a custom expression to create a new Image."""
        def parse_ternary(expr):
            while '?' in expr and ':' in expr:
                # Find the innermost ternary operation
                match = re.search(r'\(([^()]+)\)', expr)
                if match:
                    inner_expr = match.group(1)
                else:
                    inner_expr = expr

                if '?' in inner_expr and ':' in inner_expr:
                    condition, rest = inner_expr.split('?', 1)
                    true_value, false_value = rest.split(':', 1)
                    condition = condition.replace(' and ', ' & ').replace(' or ', ' | ')
                    replacement = f"np.where({condition.strip()}, {true_value.strip()}, {false_value.strip()})"
                    if match:
                        expr = expr.replace(f"({inner_expr})", replacement)
                    else:
                        expr = replacement
                else:
                    break
            return expr

        # Replace common function names with numpy equivalents
        numpy_expression = expression.replace('sqrt', 'np.sqrt')
        numpy_expression = parse_ternary(numpy_expression)
        
        numpy_vars = {key: value.data.squeeze() for key, value in variables.items()}
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            result = eval(numpy_expression, {'np': np}, numpy_vars)
        
        new_image = Image(None)
        new_image.data = np.expand_dims(result, axis=0)
        new_image.meta = next(iter(variables.values())).meta.copy()
        new_image.meta.update({'count': 1, 'dtype': result.dtype})
        new_image.crs = next(iter(variables.values())).crs
        new_image.transform = next(iter(variables.values())).transform
        new_image.bounds = next(iter(variables.values())).bounds
        new_image.nodata = next(iter(variables.values())).nodata
        new_image.band_descriptions = ['expression_result']
        
        return new_image












































