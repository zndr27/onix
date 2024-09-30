from pathlib import Path
import SimpleITK as sitk


class MapOverlay():
    """ """

    _alpha = 255
    _range = [0, 1000]
    _cmax = 1000
    _cmid = 500

    _array: np.ndarray
    _image: itk.Image
    _slicer: np.ndarray

    scaling_list = [
        "None",
        "Log",
    ]
    scaling: str = "None"

    operation_list: list[str] = [
        "None",
        "Divide",
        "Difference",
        "SSIM",
    ]
    operation: str = "None"

    def __init__(
        self,
    ):
        self._volume_1 = None
        self._volume_2 = None
        self._mask = None
        self._mask_lower = None
        self._mask_upper = None
        self._array = None
        self._image = None
        self._mask_array = None
        self._mask_image = None
        self._fixed_range = None

    def update_colormap(self, color_range, alpha, fixed_range):
        """ """
        if fixed_range is None:
            self._fixed_range = None
        else:
            self._fixed_range = fixed_range
        self._range = color_range
        self._alpha = alpha

    def get_colormap(self):
        """ """
        if self.difference:
            cs = px.colors.get_colorscale("RdBu")
        else:
            #cs = px.colors.get_colorscale("Jet")
            cs = px.colors.get_colorscale("Turbo")
        cs = px.colors.sample_colorscale(cs, np.arange(0, 254) / 254)
        cs = [
            list(map(int, x.replace("rgb(", "").replace(")", "").split(", ")))
            + [
                self._alpha,
            ]
            for x in cs
        ]
        return cs

    def get_hist(self):
        """ """
        if self._slicer is None:
            return None, None, None
        else:
            hist = self._array[np.where(self._mask_array == 1)]
            return hist, self._xlo, self._xhi

    def get_value(self, x, y, z):
        """ """
        if self._array is None:
            return np.nan
        else:
            return self._array[z, y, x]

    def get_volume_1(self, x, y, z):
        """ """
        if self.volume_1 is None:
            return np.nan
        else:
            return self.volume_1.array[z, y, x]

    def get_volume_2(self, x, y, z):
        """ """
        if self.volume_2 is None:
            return np.nan
        else:
            return self.volume_2.array[z, y, x]

    def update_mask(self, mask: OnixMask | None):
        """ """
        self._mask = mask

    def update_volume_1(self, volume: OnixVolume | None):
        """ """
        self.volume_1 = volume

    def update_volume_2(self, volume: OnixVolume | None):
        """ """
        self.volume_2 = volume

    def update(self):
        """ """
        if self.volume_1 == None:
            self._slicer = None
            return

        # Load Volume 1
        array1 = self.volume_1.array
        image1 = self.volume_1.image

        # Load Volume 2
        if self.volume_2 is not None:
            array2 = self.volume_2.array
            image2 = self.volume_2.image

            if self.operation == "SSIM":
                _, array = structural_similarity(
                    self.volume_1.array,
                    self.volume_2.array,
                    data_range=1,
                    full=True,
                )
                _, S = structural_similarity(
                    sitk.GetArrayFromImage(self.volume_1.image),
                    sitk.GetArrayFromImage(self.volume_2.image),
                    win_size=21,
                    data_range=1,
                    full=True,
                )
                image = orient_array(S, self.volume_1.image)

            elif self.operation == "Difference":
                array = array1 - array2
                image = sitk.SubtractImageFilter().Execute(image1, image2)

            elif self.operation == "Divide":
                array = array1 / array2
                image = sitk.DivideImageFilter().Execute(image1, image2)

            else:
                array = array1
                image = image1
        
        self._array = array
        self._image = image

        # Load mask
        if self._mask is None:
            self._mask_array = np.ones_like(self._array)
            self._mask_image = sitk.Image(self._image)
        else:
            self._mask_array = self._mask.array
            self._mask_image = self._mask.image

        # Update mask based on thresholds
        self._mask_array = np.where(
            np.all(
                self._mask.array >= self._mask_lower, 
                self._mask.array <= self._mask_upper,
            ),
            1, 0,
        )
        self._mask_image = binary_threshold(
            self._mask.image, 
            self._mask_lower, 
            self._mask_upper,
        )

        # Rescale map based on thresholds into range between [1, 255]
        # This way 0 will be reserved for values outside the binary mask
        self.rescale_map()

        # Convert map to uint8
        self._image = cast_uint8(self._image)

        # Apply binary mask to the map
        self._image = mask_uint8_map(self._image, self._mask_image)

        # Convert sitk to ndarray for VolumeSlicer
        self._slicer = slicer_array(self._image)

    def rescale_map(self):
        """ """
        cmid = self._cmid
        cmax = self._cmax
        lo, hi = self._range

        array = self._array
        image = self._image
        mask_array = self._mask_array
        mask_image = self._mask_image
        
        assert mask_array.shape == array.shape:

        hist = array[np.where(mask_array == 1)]
        xmin = np.min(hist)
        xmax = np.max(hist)

        R = sitk.RescaleIntensityImageFilter()

        if self.operation == "Difference":
            xrm = max(-xmin, xmax)
            xhi = xrm * abs(hi - cmid) / cmid
            xlo = xrm * abs(cmid - lo) / cmid
            xr = min(xhi, xlo)
            array = np.where(array < xr, array, xr)
            array = np.where(array > -xr, array, -xr)
            image = threshold_above(image, xr)
            image = threshold_below(image, -xr)
            _min = np.min(array)
            _max = np.max(array)
            R.SetOutputMinimum(128 + 127 * (_min / xr))
            R.SetOutputMaximum(128 + 127 * (_max / xr))
            self._xlo = -xr
            self._xhi = xr
        else:
            if self._fixed_range is not None:
                xmin = self._fixed_range[0]
                xmax = self._fixed_range[1]
            xrange = xmax - xmin
            xhi = xmin + xrange * (hi / cmax)
            xlo = xmin + xrange * (lo / cmax)
            xr = xhi - xlo
            array = np.where(array < xhi, array, xhi)
            array = np.where(array > xlo, array, xlo)
            image = threshold_above(image, xhi)
            image = threshold_below(image, xlo)
            _min = np.min(array)
            _max = np.max(array)
            R.SetOutputMinimum(1 + 254 * ((_min - xlo) / xr))
            R.SetOutputMaximum(1 + 254 * ((_max - xlo) / xr))
            self._xlo = xlo
            self._xhi = xhi

        self._array = array
        self._image = R.Execute(image)


class MaskOverlay():
    """ """
    _alpha = 255
    _range = [0, 255]
    _cmax = 255
    _cmid = 127

    _array: np.ndarray
    _image: itk.Image
    _slicer: np.ndarray

    operation_list: list[str] = [
        "None",
        "Divide",
        "Difference",
        "SSIM",
    ]
    operation: str = "None"
    _alpha = 255
    _range = [0, 1000]
    _cmax = 1000
    _cmid = 500

    _array: np.ndarray
    _image: itk.Image
    _slicer: np.ndarray

    scaling_list = [
        "None",
        "Log",
    ]
    scaling: str = "None"

    operation_list: list[str] = [
        "None",
        "Divide",
        "Difference",
        "SSIM",
    ]
    operation: str = "None"

    def __init__(
        self,
    ):
        self._mask_1 = None
        self._mask_2 = None
        self._mask = None
        self._array = None
        self._image = None
        self._fixed_range = None

    def update(self):
        """ """
        array1 *= 75
        array2 *= 150
        image1 = self.multiply_filter(image1, 75)
        image2 = self.multiply_filter(image2, 150)
        
        self._xlo = np.min(self._array)
        self._xhi = np.max(self._array)


class OnixViewer():
    """ """
    _PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
    _DBC_CSS = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css"
    _EXTERNAL_STYLESHEETS = [dbc.themes.DARKLY, _DBC_CSS]

    app: Dash

    slicer_z: VolumeSlicer
    slicer_y: VolumeSlicer
    slicer_x: VolumeSlicer

    datasets: dict[str, OnixDataset]
    dataset_pairs: list[tuple[str, str]]
    t1_tx = dict[tuple[str, str], sitk.Transform]
    ref_tx = dict[tuple[str, str], sitk.Transform]

    metrics: OnixMetrics
    overlay: OnixOverlay

    def __init__(
        self,
        datasets: dict[str, OnixDataset],
        dataset_pairs: list[tuple[str, str]],
        save_path: str | Path,
        load_path: str | Path | None,
        log: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    ):
        """
        """
        self.datasets = datasets
        self.dataset_pairs = dataset_pairs
        self.save_path = Path(save_path) 
        self.load_path = Path(load_path) if load_path is not None else None
        self.log = log

        if self.load_path is not None:
            try: 
                self._load(self.load_path)
                print('Loaded cached data...')
                return
            except Exception as e:
                print("Performing analysis...")
                
        os.makedirs(self.save_path, exist_ok=False)

        self._maps = {}
        self._masks = {}
        for name, ds in datasets.items():
            self._maps[name] = ds._maps
            self._masks[name] = ds._masks

        self._register_datasets()

        self.metrics = Metrics()
        self.map_overlay = MapOverlay()
        self.mask_overlay = MaskOverlay()

        self.app = Dash(
            __name__,
            external_stylesheets=self._EXTERNAL_STYLESHEETS,
            use_pages=True,
            pages_folder="",
            #suppress_callback_exceptions=True,
        )

        self.init_slicer()
        self.init_pages()
        self.init_callbacks()

    def _register_datasets(self):
        """
        Register the T1 and reference images for each dataset.
        """
        for a, b in self.dataset_pairs:
            dataset_1 = self.datasets[a]
            dataset_2 = self.datasets[b]

            _, self.t1_tx = register_routine(
                dataset_1.t1.image,
                dataset_2.t1.image,
                log = self.log == "DEBUG",
            )
        
            _, self.ref_tx = register_routine(
                self.fitt_ds.ref.image,
                self.nnfit_ds.ref.image,
                log = self.log == "DEBUG",
            )

    def update_overlay_volume(self, dataset_name, label, primary=False):
        """ 
        Update the overlay volume.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset.
        label : str
            The label of the volume.
        primary : bool
            Whether we update the primary or secondary volume.

        Returns
        -------
        None
        """
        ds = self.datasets[dataset_name]

        # Load volume and align to T1 image.
        volume = ds.load(label).align(ds.t1)

        if primary:
            self.overlay.update_volume_1(volume)
        else:
            self.overlay.update_volume_2(volume)

    def update_scaling(self, scaling):
        """ """
        self.overlay.update_scaling(scaling)

    def update_operation(self, operation):
        """ """
        self.overlay.update_operation(operation)

    def update_colormap(self, color_range: tuple, alpha: float, fixed_range: tuple):
        """ """
        self.overlay.update_colormap(color_range, alpha, fixed_range)

    def update_mask(self, mask: str):
        """ """
        self.overlay.update_mask(mask)

    def update_cho_naa_threshold(self, fitt_threshold: float, nnfit_threshold: float, use_avg=False):
        """ """
        masks = self.metrics.get_cho_naa_masks(
            fitt_threshold = fitt_threshold,
            nnfit_threshold = nnfit_threshold,
            use_avg = use_avg,
        )
        self.fitt_cho_naa_mask = masks.get('fitt_cho_naa_mask')
        self.nnfit_cho_naa_mask = masks.get('nnfit_cho_naa_mask')
        #self.fitt_cho_naa_mask_1connect = masks.get('fitt_cho_naa_mask_1connect')
        #self.nnfit_cho_naa_mask_1connect = masks.get('nnfit_cho_naa_mask_1connect')

    def update_overlay(self):
        """ """
        _overlay = self.overlay.get_slicer(update=True)
        _colormap = self.overlay.get_colormap()
        return (
            self.slicer_z.create_overlay_data(_overlay, _colormap),
            self.slicer_y.create_overlay_data(_overlay, _colormap),
            self.slicer_x.create_overlay_data(_overlay, _colormap),
        )

    def mrsi_to_mri_coords(self, dataset_name: str, x: float, y: float, z: float):
        """ """
        ds = self.datasets[dataset_name]
        return ds.t1.image.TransformPhysicalPointToIndex(
            ds.ref.image.TransformIndexToPhysicalPoint([x, y, z])
        )

    def mri_to_mrsi_coords(self, dataset_name: str, x: float, y: float, z: float):
        """ """
        ds = self.datasets[dataset_name]
        return ds.ref.image.TransformPhysicalPointToIndex(
            ds.t1.image.TransformIndexToPhysicalPoint([x, y, z])
        )

    def transform_coords(self, dataset_1: str, dataset_2: str, x: float, y: float, z: float):
        """ """
        ds_1 = self.datasets[dataset_1]
        ds_2 = self.datasets[dataset_2]
        return ds_2.ref.image.TransformPhysicalPointToIndex(
            list(
                self.ref_tx.TransformPoint(
                    list(
                        ds_1.ref.image.TransformIndexToPhysicalPoint([x, y, z])
                    )
                )
            )
        )

    def init_slicer(self):
        """ """
        array = self.fitt_ds.t1.slicer_array()

        self.slicer_z = VolumeSlicer(
            self.app,
            array,
            axis=0,
            thumbnail=False,
        )
        self.slicer_y = VolumeSlicer(
            self.app,
            array,
            axis=1,
            thumbnail=False,
        )
        self.slicer_x = VolumeSlicer(
            self.app,
            array,
            axis=2,
            thumbnail=False,
        )

        # Turn off scroll zoom in slicers.
        self.slicer_z.graph.config["scrollZoom"] = False
        self.slicer_y.graph.config["scrollZoom"] = False
        self.slicer_x.graph.config["scrollZoom"] = False

        # Give the slicers a dark theme.
        self.slicer_z.graph.figure.update_layout(template="plotly_dark")
        self.slicer_y.graph.figure.update_layout(template="plotly_dark")
        self.slicer_x.graph.figure.update_layout(template="plotly_dark")

    def init_pages(self):
        """ """
        from onix.layouts import main_layout, metrics_layout, slicer_layout

        dash.register_page("metrics", path="/metrics", layout=metrics_layout(self))
        dash.register_page("slicer", path="/", layout=slicer_layout(self))

        self.app.layout = main_layout(self)

    def init_callbacks(self):
        """ """
        from onix.callbacks import metrics_callbacks, slicer_callbacks

        metrics_callbacks(self)
        slicer_callbacks(self)

    def start(
        self,
        debug=True,
        dev_tools_props_check=False,
        host="0.0.0.0",
        port=8050,
        **kwargs,
    ):
        self.app.run_server(
            debug=debug,
            dev_tools_props_check=dev_tools_props_check,
            host=host,
            port=port,
            **kwargs,
        )
