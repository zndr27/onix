from abc import ABC


class OnixDataset(OnixObject):
    """ """

    t1: OnixVolume
    ref: OnixVolume
    spectra: OnixSpectra

    def __init__(self):
        pass


class FITTDataset(OnixDataset):
    """
    Represents the data from a MIDAS study.

    Attributes
    ----------
    study : MidasStudy
        The study node from MIDAS.
    study_type : str
        Whether to use the data from FITT or NNFit.
    flip_x : bool
        Whether to flip the x-axis of the data.
    log : bool
        Whether to log the registration process.
    save_outputs : bool
        Whether to save the outputs of the imported data processing.
    t1 : OnixVolume
        The T1-weighted image.
    ref : OnixVolume
        The SI reference image.
    brain_mask : OnixVolume
        The brain mask.
    qmap : OnixVolume
        The MIDAS quality mask.
    hqmap : OnixVolume
        Eroded MIDAS quality mask.
    vhqmap : OnixVolume
        2x eroded MIDAS quality mask.
    nnqmap : OnixVolume
        The neural network quality mask.
    nnhqmap : OnixVolume
        Eroded neural network quality mask.
    nnvhqmap : OnixVolume
        2x eroded neural network quality mask.
    t2star : OnixVolume
        The T2* image.
    flair : OnixVolume
        The FLAIR image.
    seg_masks : dict[str, MidasFrame]
        The segmentation masks.
    spectra : OnixSpectra
        The SI spectra.
    baseline : OnixSpectra
        The baseline spectra.
    fit : OnixSpectra
        The fit spectra.
    si_maps : dict[str, MidasFrame]
        The SI maps.
    sinorm_maps : dict[str, MidasFrame]
        The SI normalized maps.
    nnfit_ds : NNFitDataset
        The nnfit dataset.
    nnfit_maps : list[str]
        The nnfit maps.
    """
    # ^ sphnix documentation

    study: MidasStudy
    study_type: str
    flip_x: bool
    log: bool
    save_outputs: bool

    t1: OnixVolume
    ref: OnixVolume
    brain_mask: OnixVolume
    qmap: None | OnixVolume
    hqmap: None | OnixVolume
    vhqmap: None | OnixVolume
    nnqmap: None | OnixVolume
    nnhqmap: None | OnixVolume
    nnvhqmap: None | OnixVolume
    t2star: None | OnixVolume
    flair: None | OnixVolume
    seg_masks: None | dict[str, MidasFrame]

    spectra: OnixSpectra
    baseline: OnixSpectra
    fit: OnixSpectra

    si_maps: None | dict[str, MidasFrame]
    sinorm_maps: None | dict[str, MidasFrame]

    nnfit_ds: None | NNFitDataset
    nnfit_maps: None | list[str]

    def __init__(
        self, 
        study: MidasStudy, 
        study_type: str, 
        flip_x=False, 
        log=False, 
        save_outputs=True,
        og=False,
    ):
        self.study = study
        self.study_type = study_type
        self.flip_x = flip_x

        self.t1 = OnixVolume(*self.study.t1())
        self._ref = OnixVolume(*self.study.ref())
        self._brain_mask = OnixVolume(*self.study.brain_mask())

        self._ref = self.flip(self._ref)
        self._brain_mask = self.flip(self._brain_mask)

        self.ref = copy.deepcopy(self._ref)
        self.brain_mask = copy.deepcopy(self._brain_mask)

        # TODO
        onix_path = study.subject_path / 'onix' / study.date.replace('/','_')
        self.onix_path = onix_path

        if save_outputs:

            print("DATASET REGISTER ROUTINE")

            _, self.tx = _register_routine(
                self.t1.image,
                self.ref.align(self.t1).image,
                learn_rate=0.01,
                stop=0.001,
                max_steps=50,
                log=log,
            )

            with open(onix_path / (study_type + '_tx.pkl'), 'wb') as f:
                pickle.dump(self.tx, f, pickle.HIGHEST_PROTOCOL)
        else:

            print("LOADING DATASET REGISTER ROUTINE")

            with open(onix_path / (study_type + '_tx.pkl'), 'rb') as f:
                self.tx = pickle.load(f)

        self.ref = self.apply_tx(self.ref)
        self.brain_mask = self.apply_tx(self.brain_mask)

        if self.study_type == "fitt":
            try:
                self.flair = OnixVolume(*self.study.flair())
            except Exception as e:
                print(f"\nWARNING: unable to load Flair series\n{e}\n")
                self.flair = None
            try:
                self.seg_masks = {
                    x.param("Frame_Type"): x
                    for x in self.study.process("MRI_SEG").dataset("MriSeg").all_frame()
                }
            except Exception as e:
                print(f"\nWARNING: unable to load segmentation masks\n{e}\n")
                self.seg_masks = None

            self.qmap = OnixVolume(*self.study.qmap())
            self.t2star = OnixVolume(*self.study.t2star())

            self.qmap = self.apply_tx(self.flip(self.qmap))
            self.t2star = self.apply_tx(self.flip(self.t2star))

            assert self.study.si() is not None
            assert self.study.fitt() is not None
            #assert self.study.fitt_baseline() is not None
            self.spectra = OnixSpectra(self.study.si())
            self.baseline = OnixSpectra(self.study.fitt_baseline())
            self.fit = OnixSpectra(self.study.fitt())
            # self.fit     = OnixSpectra(self.study.fitt() - self.study.fitt_baseline())

            self.si_maps = {
                x.param("Frame_Type"): x
                for x in self.study.series("SI")
                .process("Maps")
                .input("FITT")
                .data()
                .all_frame()
            }
            self.sinorm_maps = {
                x.param("Frame_Type"): x
                for x in self.study.series("SI")
                .process("Maps")
                .data("SINorm")
                .all_frame()
            }

            self.nnfit_ds = None
            self.nnfit_maps = None

            _erosion_time = time.time()

            # TODO qmap erosion >>>
            _hqmap = binary_erosion(self.qmap.array == 4).astype(np.float32)
            _vhqmap = binary_erosion(_hqmap).astype(np.float32)
            self.hqmap = OnixVolume(_hqmap, orient_array(_hqmap, self.brain_mask.image))
            self.vhqmap = OnixVolume(
                _vhqmap, orient_array(_vhqmap, self.brain_mask.image)
            )
            # TODO <<<

            print("erosion time 1", time.time() - _erosion_time)

            # TODO nnqmap >>>
            import tensorflow as tf

            _qmap_time = time.time()

            _arg_mask = np.where(self.brain_mask.array.reshape(-1) == 1)[0]
            _spectra = self.spectra.array.reshape((-1, 512))[_arg_mask]
            model = tf.saved_model.load(
                f"/workspace/Dropbox/nn/nnfit/models/qc_test_2/EPOCH.3/saved_model/"
            )
            _latent, _label = model(_spectra.real, training=True)
            _label = np.argmax(tf.nn.softmax(_label), axis=1)
            _arg_nnqmap = _arg_mask[np.where(_label == 1)]
            _nnqmap = np.zeros(self.brain_mask.array.shape)
            _nnqmap[np.unravel_index(_arg_nnqmap, _nnqmap.shape)] = 1
            print("qmap time", time.time() - _qmap_time)

            _erosion_time_2 = time.time()

            _nnhqmap = binary_erosion(_nnqmap).astype(np.float32)
            _nnvhqmap = binary_erosion(_nnhqmap).astype(np.float32)

            print("erosion time 2", time.time() - _erosion_time_2)

            self.nnqmap = OnixVolume(
                _nnqmap, orient_array(_nnqmap, self.brain_mask.image)
            )
            self.nnhqmap = OnixVolume(
                _nnhqmap, orient_array(_nnhqmap, self.brain_mask.image)
            )
            self.nnvhqmap = OnixVolume(
                _nnvhqmap, orient_array(_nnvhqmap, self.brain_mask.image)
            )
            # TODO <<<

        elif self.study_type == "nnfit":
            self.nnfit_ds = NNFitDataset(self.study, og=og)

            assert self.nnfit_ds.load_spectra() is not None
            assert self.nnfit_ds.load_baseline() is not None
            self.spectra = OnixSpectra(self.nnfit_ds.load_spectra())
            self.baseline = OnixSpectra(self.nnfit_ds.load_baseline())
            self.fit = OnixSpectra(self.baseline.array + self.nnfit_ds.load_peaks())
            # self.fit     = OnixSpectra(self.nnfit_ds.load_peaks())

            if og:
                self.og_maps = self.nnfit_ds.og_maps
                self._og = self.nnfit_ds._og
                self.og_spectra = None
                self.og_baseline = None
                self.og_fit = None

            if self.flip_x:
                self.spectra = self.spectra.flip_x()
                self.baseline = self.baseline.flip_x()
                self.fit = self.fit.flip_x()

            self.nnfit_maps = ["Ta", "Tb", "dw", "phi0"]
            self.nnfit_maps += [f"{x} area" for x in list(self.nnfit_ds.metabolite)]
            self.nnfit_maps += [f"{x} shift" for x in list(self.nnfit_ds.metabolite)]
            self.nnfit_maps += [f"{x} ratio" for x in list(self.nnfit_ds.ratio)]
            
            if 'cho area' in self.nnfit_maps:
                self.nnfit_maps += [f"cho area x3",]

            self.si_maps = None
            self.sinorm_maps = None

            self.ppm = self.nnfit_ds.ppm

        elif self.study_type == "extra":
            pass

        else:
            raise Exception(f"Invalid study type provided: {self.study_type}")

    def flip(self, vol) -> OnixVolume:
        """ """
        if self.flip_x and self.study_type == "nnfit":
            return vol.flip_x()
        else:
            return vol

    def apply_tx(self, vol) -> OnixVolume:
        """ """
        vol = vol.align(self.t1).register(self.t1, self.tx).align(self.ref)
        vol.array = itk.GetArrayFromImage(vol.image)
        return vol

    def spec_to_ref_coords(self, x, y, z):
        """ """
        return self.ref.image.TransformPhysicalPointToIndex(
            list(
                self.tx.TransformPoint(
                    list(self.ref.image.TransformIndexToPhysicalPoint([x, y, z]))
                )
            )
        )

    def ref_to_spec_coords(self, x, y, z):
        """ """
        return self.ref.image.TransformPhysicalPointToIndex(
            list(
                self.tx.GetInverse().TransformPoint(
                    list(self.ref.image.TransformIndexToPhysicalPoint([x, y, z]))
                )
            )
        )

    def load_si_map(self, frame_type) -> OnixVolume:
        """ """
        if self.si_maps == None:
            return None
        return self.apply_tx(self.flip(OnixVolume(*self.si_maps[frame_type].load())))

    def load_sinorm_map(self, frame_type) -> OnixVolume:
        """ """
        if self.sinorm_maps == None:
            return None
        return self.apply_tx(
            self.flip(OnixVolume(*self.sinorm_maps[frame_type].load()))
        )

    def load_seg_mask(self, frame_type) -> OnixVolume:
        """ """
        if self.seg_masks == None:
            return None
        return self.apply_tx(self.flip(OnixVolume(*self.seg_masks[frame_type].load())))

    def load_nnfit_map(self, label) -> OnixVolume:
        """ """
        if self.nnfit_maps == None:
            return None

        if label == 'cho area x3':
            vol = OnixVolume(*self.nnfit_ds.load_area('cho'))
            vol.array *= 3
            mul_filter = itk.MultiplyImageFilter[vol.image, vol.image, vol.image].New()
            mul_filter.SetInput(vol.image)
            mul_filter.SetConstant(3)
            vol.image = mul_filter.GetOutput()

        elif label.split()[-1] == "area":
            vol = OnixVolume(*self.nnfit_ds.load_area(label.split()[0]))

        elif label.split()[-1] == "shift":
            vol = OnixVolume(*self.nnfit_ds.load_shift(label.split()[0]))

        elif label.split()[-1] == "ratio":
            vol = OnixVolume(*self.nnfit_ds.load_ratio(label.split()[0]))

        elif label in ("Ta", "Tb", "dw", "phi0"):
            vol = OnixVolume(*self.nnfit_ds.load_map(label))

        else:
            raise Exception(f"{label} is not an nnfit map")

        return self.apply_tx(self.flip(vol))

    def phase(self, x, y, z):
        """ """
        return self.nnfit_ds.phase(x=x, y=y, z=z)



class NNFitDataset(OnixObject):
    """
    Represents the data from an nnfit process.

    Attributes
    ----------
    study : MidasStudy
        The study node from MIDAS.
    process : MidasProcess
        The nnfit process node from MIDAS.
    data : MidasData
        The nnfit data node from MIDAS.
    xr_data : MidasData
        The xarray data node from MIDAS.
    """
    # ^ sphnix documentation

    study: MidasStudy
    process: MidasProcess
    data: MidasData
    xr_data: MidasData

    def __init__(self, study: MidasStudy, og=True):
        self.study = study
        self.process = self.study.series("SI").process("nnfit")
        self.data = self.process.data("nnfit")
        self.xr_data = self.process.data("xarray")

        if og:
            self.load_og()
            self._og = True
        else:
            try:
                self.load_og()
                self._og = True
            except Exception as e:
                print(f"\nWARNING: unable to load OG data\n{e}\n")
                self._og = False

        self.metabolite = self.open_ds().metabolite.data
        self.ratio = self.open_ds().ratio.data
        self.ppm = self.open_ds().ppm.data

    def load_og(self):
        """ """
        maps_process = self.study.series("SI").process("Maps")
        self.og_data = maps_process.data("NNFIT")

        self.og_cho = OnixVolume(*self.og_data.frame("nnfit_CHO_Area").load())
        self.og_cr = OnixVolume(*self.og_data.frame("nnfit_CR_Area").load())
        self.og_naa = OnixVolume(*self.og_data.frame("nnfit_NAA_Area").load())
        self.og_cho_naa = OnixVolume(*self.og_data.frame("nnfit_CHO/NAA").load())

        self.og_maps = ['og_cho', 'og_cr', 'og_naa', 'og_cho_naa']

        nnfit_dir = self.study.subject_path / "nnfit"

        spec_file = nnfit_dir / self.og_data.param("nnfit_spectrum_file")
        self.og_spec = np.fromfile(spec_file, dtype=np.float32).reshape(
            int(self.og_data.param("Spatial_Points_3")),
            int(self.og_data.param("Spatial_Points_2")),
            int(self.og_data.param("Spatial_Points_1")),
            512,
        )
        
        base_file = nnfit_dir / self.og_data.param("nnfit_baseline_file")
        self.og_base = np.fromfile(base_file, dtype=np.float32).reshape(
            int(self.og_data.param("Spatial_Points_3")),
            int(self.og_data.param("Spatial_Points_2")),
            int(self.og_data.param("Spatial_Points_1")),
            512,
        )

    def load_og_map(self, label) -> (np.ndarray, itk.Image):
        """ """
        if label == "og_cho":
            return OnixVolume(self.og_cho.array, self.og_cho.image)
        elif label == "og_cr":
            return OnixVolume(self.og_cr.array, self.og_cr.image)
        elif label == "og_naa":
            return OnixVolume(self.og_naa.array, self.og_naa.image)
        elif label == "og_cho_naa":
            return OnixVolume(self.og_cho_naa.array, self.og_cho_naa.image)
        else:
            raise Exception(f"{label} is not an OG nnfit map")

    def open_ds(self):
        """
        Open the xarray dataset.
        """
        return xr.open_zarr(self.xr_data.path, decode_times=False).sel(frame="Original")

    def ndarray_to_itk(self, array: np.ndarray) -> itk.Image:
        """
        Convert a numpy array to an itk image.

        Assumes that metadata is stored in Midas data object.
        """
        # Extract the metadata for the image
        px = float(self.data.param("Image_Position_X"))
        py = float(self.data.param("Image_Position_Y"))
        pz = float(self.data.param("Image_Position_Z"))
        dx = int(self.data.param("Spatial_Points_1"))
        dy = int(self.data.param("Spatial_Points_2"))
        dz = int(self.data.param("Spatial_Points_3"))
        sx = float(self.data.param("Pixel_Spacing_1"))
        sy = float(self.data.param("Pixel_Spacing_2"))
        sz = float(self.data.param("Pixel_Spacing_3"))
        oxr = float(self.data.param("Image_Orientation_Xr"))
        oyr = float(self.data.param("Image_Orientation_Yr"))
        ozr = float(self.data.param("Image_Orientation_Zr"))
        oxc = float(self.data.param("Image_Orientation_Xc"))
        oyc = float(self.data.param("Image_Orientation_Yc"))
        ozc = float(self.data.param("Image_Orientation_Zc"))

        # Pass the raw data to an itk image
        image = itk.GetImageFromArray(array)

        # Update image with the metadata
        image.SetSpacing([sx, sy, sz])
        image.SetOrigin(update_origin(px, py, pz, oxr, oyr, ozr, oxc, oyc, ozc))
        image.SetDirection(
            np.array(
                [
                    [oxr, oyr, ozr],
                    [oxc, oyc, ozc],
                    cross_product(oxr, oyr, ozr, oxc, oyc, ozc),
                ],
                dtype=np.float64,
            )
        )

        # Cast image to float, since image operations are float
        if array.dtype != np.float32:
            image = cast_float(image)

        return image

    def load_area(self, metabolite: str) -> (np.ndarray, itk.Image):
        """
        Load metabolite area (i.e. amplitude) map.
        """
        ds = self.open_ds()
        array = ds.sel(metabolite=metabolite).areas.data.compute()
        image = self.ndarray_to_itk(array)
        return array, image

    def load_shift(self, metabolite: str) -> (np.ndarray, itk.Image):
        """
        Load metabolite-specific frequency shift map.
        """
        ds = self.open_ds()
        array = (
            ds.dw.data.compute() + ds.sel(metabolite=metabolite).shifts.data.compute()
        )
        image = self.ndarray_to_itk(array)
        return array, image

    def shift(self, x, y, z, option="points"):
        """
        Load the frequency shift (based on NAA) at the specified coordinates.
        """
        ds = self.open_ds().sel(z=z, y=y, x=x)
        freq_shift = (
            ds.dw.data.compute() + ds.sel(metabolite="naa").shifts.data.compute()
        )

        if option == "points":
            hz_per_pt = self.study.si_sampling().get("hz_per_pt")
            return int(freq_shift / (2 * np.pi) / hz_per_pt)
        elif option == "ppm":
            hz_per_ppm = self.study.si_sampling().get("hz_per_ppm")
            return freq_shift / (2 * np.pi) / hz_per_ppm
        elif option == "hz":
            return freq_shift / (2 * np.pi)
        else: 
            # rad/s
            return freq_shift

    def phase(self, x, y, z):
        """
        Load the phase at the specified coordinates.
        """
        ds = self.open_ds()
        return ds.phi0.sel(z=z, y=y, x=x).data.compute()

    def load_map(self, label) -> (np.ndarray, itk.Image):
        """
        Load the specified map.
        """
        ds = self.open_ds()
        array = ds.get(label).data.compute()
        image = self.ndarray_to_itk(array)
        return array, image

    def load_ratio(self, ratio) -> (np.ndarray, itk.Image):
        """
        Load the specified ratio map.
        """
        ds = self.open_ds()
        array = ds.sel(ratio=ratio).ratios.data.compute()
        image = self.ndarray_to_itk(array)
        return array, image

    def load_spectra(self) -> np.ndarray:
        """
        Load the spectral data.
        """
        ds = self.open_ds()
        return ds.spectrum.data.compute()

    def load_peaks(self):
        """
        Load the peaks fit (i.e. metabolite signal fit).
        """
        ds = self.open_ds()
        return ds.peaks.data.compute()

    def load_baseline(self):
        """
        Load the baseline fit.
        """
        ds = self.open_ds()
        return ds.baseline.data.compute()
