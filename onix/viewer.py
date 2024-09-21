from onix.objects import OnixObject


class OnixViewer(OnixObject):
    """ """

    _PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
    _DBC_CSS = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css"
    _EXTERNAL_STYLESHEETS = [dbc.themes.DARKLY, _DBC_CSS]

    subject_id: str
    study_date: str

    app: Dash

    slicer_z: VolumeSlicer
    slicer_y: VolumeSlicer
    slicer_x: VolumeSlicer

    fitt_ds: OnixDataset
    nnfit_ds: OnixDataset
    ds: dict[str, OnixDataset]

    t1_tx = None
    ref_tx = None

    metrics: OnixMetrics

    overlay_list: list[str]
    overlay: OnixOverlay

    _cache_list: list[str] = [
        'fitt_tx',
        'nnfit_tx',
        't1_tx',
        'ref_tx',
        'map_df',
        'map_df_t1',
        'map_table',
        'map_table_t1',
        'ratio_df',
        'ratio_table',
        'ccb_mask',
    ]

    def __init__(
        self,
        datasets: dict[str, OnixDataset],
        save_path: str | Path,
        load_path: str | Path | None,
    ):
        """
        """
        self.datasets = datasets
        self.save_path = Path(save_path) 
        self.load_path = Path(load_path) if load_path is not None else None

        if self.load_path is not None:
            try: 
                self._load_cache()
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

        self._register_routine()

        if save_outputs:

            print("REGISTER ROUTINES")

            _, self.t1_tx = _register_routine(
                self.fitt_ds.t1.image,
                self.nnfit_ds.t1.image,
                learn_rate=0.01,
                stop=0.001,
                max_steps=50,
                log=log,
            )

            with open(onix_path / 't1_tx.pkl', 'wb') as f:
                pickle.dump(self.t1_tx, f, pickle.HIGHEST_PROTOCOL)

            _, self.ref_tx = _register_routine(
                self.fitt_ds.ref.image,
                self.nnfit_ds.ref.image,
                learn_rate=0.01,
                stop=0.001,
                max_steps=50,
                log=log,
            )

            with open(onix_path / 'ref_tx.pkl', 'wb') as f:
                pickle.dump(self.ref_tx, f, pickle.HIGHEST_PROTOCOL)

        else:

            print("LOADING REGISTRATION ROUTINES") 

            with open(onix_path / 't1_tx.pkl', 'rb') as f:
                self.t1_tx = pickle.load(f)
            with open(onix_path / 'ref_tx.pkl', 'rb') as f:
                self.ref_tx = pickle.load(f)

        print("time 2:", _profile_time - time.time())
        _profile_time = time.time()

        ###self.nnfit_ds.spectra  = self.nnfit_ds.spectra.register(self.fitt_ds.ref, self.nnfit_ds.ref, self.ref_tx)
        ###self.nnfit_ds.baseline = self.nnfit_ds.baseline.register(self.fitt_ds.ref, self.nnfit_ds.ref, self.ref_tx)
        ###self.nnfit_ds.fit      = self.nnfit_ds.fit.register(self.fitt_ds.ref, self.nnfit_ds.ref, self.ref_tx)

        print("time 3:", _profile_time - time.time())
        _profile_time = time.time()

        _metrics_time = time.time()
        self.metrics = OnixMetrics(
            self.fitt_ds,
            self.nnfit_ds,
            self.t1_tx,
            self.ref_tx,
            self.fitt_ds.brain_mask,
            self.fitt_ds.qmap,
            save_outputs = save_outputs,
            overwrite_ccb = overwrite_ccb,
        )
        print("metrics time:", _metrics_time - time.time())

        # Initialize cho/naa masks
        self.update_cho_naa_threshold(2.0, 2.0)

        _overlay_time = time.time()
        self.overlay = OnixOverlay(
            ref=self.fitt_ds.ref,
            brain_mask=self.fitt_ds.brain_mask.align(self.fitt_ds.t1),
            qmap=self.fitt_ds.qmap.align(self.fitt_ds.t1),
            hqmap=self.fitt_ds.hqmap.align(self.fitt_ds.t1),
            vhqmap=self.fitt_ds.vhqmap.align(self.fitt_ds.t1),
            nnqmap=self.fitt_ds.nnqmap.align(self.fitt_ds.t1),
            nnhqmap=self.fitt_ds.nnhqmap.align(self.fitt_ds.t1),
            nnvhqmap=self.fitt_ds.nnvhqmap.align(self.fitt_ds.t1),
            t2star=self.fitt_ds.t2star.align(self.fitt_ds.t1),
            ccb_mask=self.metrics.ccb_mask,
            volume_1=None,
            volume_2=None,
        )
        # From metrics...
        self.overlay_list += ['fitt cho/naa mask', 'nnfit cho/naa mask']
        #self.overlay_list += ['fitt cho/naa mask connect', 'nnfit cho/naa mask connect']
        self.overlay_list += ['fitt cho/naa 2x', 'nnfit cho/naa 2x']
        self.overlay_list += ['fitt cho/naa 2x connect', 'nnfit cho/naa 2x connect']
        self.overlay_list += ['fitt cho/naa 2x connect2', 'nnfit cho/naa 2x connect2']
        print("overlay time:", _overlay_time - time.time())

        print("time 4:", _profile_time - time.time())
        _profile_time = time.time()

        ###if jupyter:
        ###    self.app = JupyterDash(
        ###        __name__,
        ###        external_stylesheets=self._EXTERNAL_STYLESHEETS,
        ###        use_pages=True,
        ###        # suppress_callback_exceptions=True,
        ###    )
        ###else:
        ###    self.app = Dash(
        ###        __name__,
        ###        external_stylesheets=self._EXTERNAL_STYLESHEETS,
        ###        use_pages=True,
        ###        # suppress_callback_exceptions=True,
        ###    )
        self.app = Dash(
            __name__,
            external_stylesheets=self._EXTERNAL_STYLESHEETS,
            use_pages=True,
            pages_folder="",
            # suppress_callback_exceptions=True,
        )

        print("time 5:", _profile_time - time.time())
        _profile_time = time.time()

        self.init_slicer()
        self.init_pages()
        self.init_callbacks()

        print("time 6:", _profile_time - time.time())
        _profile_time = time.time()

        # SAVE METRICS
        if save_metrics_path is not None:
            os.makedirs(save_metrics_path, exist_ok=True)
            self.metrics.save_metrics(save_metrics_path)

    def _load_cache(self):
        """
        Load data from previous Onix session.
        """
        # TODO
        ### files = list(x.name for x in onix_path.iterdir())
        ### missing = (x for x in self._cache_list if x not in files)
        ### if any(missing):
        ###     raise Exception("Cache missing files")
        pass

    def _register_routine(self):
        """
        Register the T1 and reference images for each dataset.
        """
        pass

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

    def si_to_mri_coords(self, dataset_name: str, x: float, y: float, z: float):
        """ """
        ds = self.datasets[dataset_name]
        return ds.t1.image.TransformPhysicalPointToIndex(
            ds.ref.image.TransformIndexToPhysicalPoint([x, y, z])
        )

    def mri_to_si_coords(self, dataset_name: str, x: float, y: float, z: float):
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
