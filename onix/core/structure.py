import datetime

from onix.data import MRI, MRSI, Mask, Volume
from onix.utils.data import rigid_registration


class Analysis():
    """ """
    session: Session

    def __init__(self, session: Session):
        """ """
        self.session = session

class MRSI_Analysis(Analysis):
    fits: dict[str, MRSI]
    maps: dict[str, Volume]
    masks: dict[str, Mask]

    def __init__(self, session: Session):
        """ """
        super().__init__(session)
        self.fit = {}
        self.maps = {}
        self.masks = {}


class MRI_Analysis(Analysis):
    maps: dict[str, Volume]
    masks: dict[str, Mask]

    def __init__(self, session: Session):
        """ """
        super().__init__(session)
        self.maps = {}
        self.masks = {}


class Session(ABC):
    """ """
    id: str
    subject: Subject
    dataset: Dataset

    date: datetime.date

    mri: MRI
    mrsi: MRSI
    mrsi_ppm: np.ndarray
    mrsi_ref: Volume
    
    mri_analyses: dict[str, MRI_Analysis]
    mrsi_analyses: dict[str, MRSI_Analysis]

    def __init__(self):
        self._register_ref_to_t1()

    def _save(self):
        """ """
        pass

    def _register_ref_to_t1(self):
        """
        Register the SI reference image to the T1 image, and apply to all other SI volumes.
        """
        _, self.tx = rigid_registration(
            self.t1.image, 
            self.ref.align(self.t1).image,
            log = self.log == "DEBUG",
        )

        self.ref = apply_tx(self.ref)

        for analysis in self.mrsi_analyses.values():
            for name, vol in analysis.maps.items():
                analysis.maps[name] = apply_tx(vol)
            for name, vol in analysis.masks.items():
                analysis.masks[name] = apply_tx(vol)

    def apply_tx(self, vol: OnixVolume) -> OnixVolume:
        """ """
        vol = vol.align(self.t1).register(self.t1, self.tx).align(self.ref)
        vol.array = sitk.GetArrayFromImage(vol.image)
        return vol

    def _spec_to_ref_coords(self, x: int, y: int, z: int):
        """
        Convert original spectral coordinates to the registred SI reference image coordinates.
        """
        return self.ref.image.TransformPhysicalPointToIndex(
            list(
                self.tx.TransformPoint(
                    list(self.ref.image.TransformIndexToPhysicalPoint([x, y, z]))
                )
            )
        )

    def _ref_to_spec_coords(self, x: int, y: int, z: int):
        """
        Convert registred SI reference image coordinates to the original spectral coordinates.
        """
        return self.ref.image.TransformPhysicalPointToIndex(
            list(
                self.tx.GetInverse().TransformPoint(
                    list(self.ref.image.TransformIndexToPhysicalPoint([x, y, z]))
                )
            )
        )


class Subject():
    """ """
    id: str
    dataset: Dataset
    sessions = dict[str, Session]

    def __init__(self, subject_id: str, datset_id: str, sessions: list[Session]):
        """ """
        self.subject_id = subject_id
        self.dataset_id = dataset_id

        self.sessions = sessions

    def _save(self, save_path: Path):
        """ """
        path = save_path / self.subject_id
        os.makedirs(path, exist_ok=True)
        for session in self.sessions:
            session._save(path)


class Dataset():
    """ """
    id: str
    subjects = dict[str, Subject]

    def __init__(self, dataset_id: str, subjects: list[Subject]):
        """ """
        self.dataset_id = dataset_id
        self.subjects = subjects

    def _save(self, save_path: Path):
        """ """
        path = save_path / self.dataset_id
        os.makedirs(path, exist_ok=True)
        for subject in self.subjects:
            subject._save(path)
