from onix.data import Volume, Mask, MRI, MRSI
from onix.structure import MriAnalysis, MrsiAnalysis, Session


class MidasSession(Session):
    """ """
    def __init__(
        self, 
        subject_xml: Path | str, 
        study_date: date.time | str | None,
    ):
        # Private attributes for navigating MIDAS subject.xml
        self._subject_xml = Path(subject_xml)
        self._study_date = date.time(study_date).format("%m/%d/%Y")
        self._subject = MidasSubject(self._subject_xml)
        self._study = self._subject.study(self._study_date)

        study = self._study

        self.mri = MRI(*study.t1())
        self.mrsi = MRSI(study.si())
        self.mrsi_ref = Volume(*study.ref())
        
        sampling = study.si_sampling()
        self.mrsi_ppm = np.linspace(
            sampling["chem_shift_ref"],
            sampling["chem_shift_ref"] - sampling["ppm_range"],
            sampling["spec_pts"]
        )

        super().__init__()


class FITT(MrsiAnalysis):
    """ """
    index = {
        "CHO_Area": "Cho",
        "CR_Area": "Cr",
        "NAA_Area": "NAA",
        "GLX_Area": "Glx",
        "MINO_Area": "mI",
        "CHO": "Cho",
        "CR": "Cr",
        "NAA": "NAA",
        "GLX": "Glx",
        "MINO": "mI",
    }

    def __init__(self, session: MidasSession):
        super().__init__(session)

        # Get the study node from the subject.xml
        study = self.session._study

        self.masks["Brain"] = Mask(*study.brain_mask()),
        self.masks["QMap"] = Mask(*study.qmap()),
        
        # Load metabolite maps
        self.maps |= {
            self.index[frame.param("Frame_Type")]: Volume(*frame.load())
            for frame in study.series("SI")
            .process("Maps")
            .input("FITT")
            .data()
            .all_frame()
            if "Area" in frame.param("Frame_Type")
            and "CramRao" not in frame.param("Frame_Type")
        }

        # Load metabolite ratio maps
        self.maps |= {
            "/".join( 
                self.index[x]
                for x in frame.param("Frame_Type").split("/") 
            ): Volume(*frame.load())
            for frame in study.series("SI")
            .process("Maps")
            .data("SINorm")
            .all_frame()
            if "/" in frame.param("Frame_Type")
        }
        
        # Load spectral fits
        fitt = study.fitt()
        self.fits["Total"] = MRSI(fitt),
        try:
            baseline = self.study.fitt_baseline()
            self.fits["Baseline"] = MRSI(baseline)
            self.fits["Metabolites"] = MRSI(fitt - baseline)
        except:
            # TODO log missing baseline fit
            pass


class MRI_SEG(MriAnalysis):
    """ """
    def __init__(self, session: MidasSession):
        super().__init__(session)

        # Get the study node from the subject.xml
        study = self.session._study

        self.masks |= {
            frame.param("Frame_Type"): Mask(*frame.load())
            for frame in study.process("MRI_SEG").dataset("MriSeg").all_frame()
        }


class NAWM(MriAnalysis):
    """ """
    def __init__(self, session: MidasSession):
        super().__init__(session)

        # Get the study node from the subject.xml
        study = self.session._study

        # Load NAWM masks
        nawm_file = study.subject_path / "mri" / (
            "*" + study.date.replace("/", "_") + "*nawm.img"
        )
        nawm_file = glob.glob(str(nawm_file))
        if nawm_file:
            nawm_file = nawm_file[0]
        else:
            return

        nawm_array = np.fromfile(nawm_file, dtype=np.uint8).reshape(
            session.mri.array.shape
        )

        nawm_image = sitk.GetImageFromArray(nawm_array)
        nawm_image = align_image(nawm_image, session.mri.image)

        self.masks["NAWM"] = Mask(nawm_array, nawm_image)


class NNFit(MrsiAnalysis):
    """ """
    index = {
        "cho": "Cho",
        "cr": "Cr",
        "naa": "NAA",
        "glx": "Glx",
        "mi": "mI",
    }

    def __init__(self, session: MidasSession):
        """ """
        super().__init__(session)

        # Get the NNFit node from the subject.xml
        study = self.session._study
        data = study.series("SI").process("nnfit").data("nnfit")

        # Open the NNFit xarray dataset
        ds = xr.open_zarr(
            study.series("SI").process("nnfit").data("xarray").path, 
            decode_times=False,
        ).sel(
            frame="Original"
        )

        # Load spectral fits
        self.fits |= {
            "Total": MRSI((ds.peaks + ds.baseline).compute().data),
            "Metabolite": MRSI(ds.peaks.data.compute()),
            "Baseline": MRSI(ds.baseline.data.compute()),
        }

        # Load metabolite maps
        for name in list(ds.metabolite):
            array = ds.sel(metabolite=name).areas.data.compute()
            image = midas_array_to_sitk(data, array)
            self.maps[self.index[name]] = Volume(array, image)

        # Load metabolite ratio maps
        for name in list(ds.ratio):
            array = ds.sel(ratio=name).ratios.data.compute()
            image = midas_array_to_sitk(data, array)
            self.maps[
                "/".join(self.index[x] for x in name.split("/"))
            ] = Volume(array, image)
