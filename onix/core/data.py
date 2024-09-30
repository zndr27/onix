import numpy as np
import SimpleITK as sitk
import skimage
from skimage.morphology import isotropic_closing
import tempfile
from typing import Literal
from typing_extensions import Self

from onix.utils import register_resample, align_map


class Volume():
    """
    Class that represents a 3D image.

    Attributes
    ----------
    array : np.ndarray
        The raw image data as a numpy array.
    image : itk.Image
        The image data as an itk.Image.
    is_mask : bool  
        Whether the volume is a mask or not.
    _silent : bool
        Whether to raise warnings or not.

    Methods
    -------
    register(fixed, tx)
        Register data to the fixed image using the specified transform.
    align(fixed)
        Align data to the coordinate system of the fixed image.
    connected_component()
        Find the largest connected component in the mask.
    slicer_array()
        Get the image data as a numpy array for the slicer.
    flip_x()
        Flip the volume along the x-axis.
    """
    # ^ sphnix documentation

    array: np.ndarray
    image: itk.Image
    is_mask: bool

    def __init__(self, array: np.ndarray, image: sitk.Image):
        self.array = array
        self.image = image

    def register(self, fixed: Self, tx) -> Self:
        """
        Register sitk data to the fixed volume using the specified transform.

        Parameters
        ----------
        fixed : OnixVolume
            The fixed image to register to.
        tx : sitk.Transform
            The transform to use for registration.
        """
        image = register_resample(fixed.image, self.image, tx)
        return OnixVolume(self.array, image)

    def align(self, fixed: Self) -> Self:
        """
        Align sitk data to the coordinate system of the fixed volume.

        Parameters
        ----------
        fixed : OnixVolume
            The fixed image to align to.

        Returns
        -------
        OnixVolume
            Volume with aligned image.
        """
        image = align_image(self.image, fixed.image)
        return OnixVolume(self.array, image)

    def slicer_array(self) -> np.ndarray:
        """
        Get the image data as a numpy array for the slicer.
        """
        return slicer_array(self.image)


class Mask(Volume):
    """
    Class that represents a 3D binary mask.
    """
    def __init__(self, array: np.ndarray, image: sitk.Image):
        super().__init__(array, image)

    def connected_component(self) -> Self:
        """
        Find the largest connected component in the mask.

        Returns
        -------
        OnixVolume
            The largest connected component.
        """
        f = isotropic_closing
        img_bw = f(self.array, 2)
        labels = skimage.measure.label(
            img_bw, 
            return_num=False,
        )
        maxCC_nobcg = labels == np.argmax(
            np.bincount(labels.flat, weights=img_bw.flat)
        )
        array = maxCC_nobcg.astype(np.uint8)
        image = sitk.GetImageFromArray(array)
        image.SetOrigin(self.image.GetOrigin())
        image.SetSpacing(self.image.GetSpacing())
        image.SetDirection(self.image.GetDirection())
        return Mask(array, image)


class MRI(Volume):
    """
    Structural MRI scan (T1, T2, or PD).

    Corresponds to 'anat' in the BIDS specification.
    """
    self.skull: OnixMask
    self.brain_mask: OnixMask

    def __init__(
        self, 
        array: np.ndarray, 
        image: sitk.Image, 
        scan_type: Literal["T1", "T2", "PD"],
    ):
        super().__init__(array, image)
        self.scan_type = scan_type
        self.brain_mask = None

    def skull_strip(self):
        """
        Skull strip the MRI to obtain the brain mask.
        """
        # Create temporary directory with context manager
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)

            # Save the MRI to the temporary directory in NIfTI format
            sitk.WriteImage(self.image, temp_dir / "mri.nii.gz")

            # Use FSL BET to skull strip the MRI
            from nipype.interfaces import fsl
            bet = fsl.BET()
            bet.inputs.in_file = temp_dir / "mri.nii.gz"
            bet.inputs.out_file = temp_dir / "brain.nii.gz"
            bet.inputs.mask = True
            res = bet.run()

            # Load the brain mask with SimpleITK
            brain_mask = sitk.ReadImage(res.outputs.mask_file)

        self.brain_mask = OnixMask(
            sitk.GetArrayFromImage(brain_mask),
            brain_mask,
        )


class MRSI(OnixObject):
    """
    Represents a 4D MR spectroscopic image.

    Assumes the data is in the shape (z, y, x, f).

    Corresponds to 'mrs' in the BIDS specification.

    Attributes
    ----------
    array : np.ndarray
        The raw image data as a numpy array.

    Methods
    -------
    flip_x()
        Flip the data along the x-axis.
    get(x, y, z, component="real", phase=None)
        Get the spectrum at the specified coordinates.
    """

    array: np.ndarray

    def __init__(self, array: np.ndarray):
        self.array = array

    def get(
        self, 
        x: int, 
        y: int, 
        z: int, 
        phase: float = 0.,
        shift: float = 0.,
    ) -> np.ndarray:
        """
        Get the spectrum at the specified coordinates.

        Parameters
        ----------
        x : int
            The x-coordinate.
        y : int
            The y-coordinate.
        z : int
            The z-coordinate.
        component : str
            The component of the spectrum to return. Options are "real", "imaginary", and "magnitude".
        phase : float
            The phase to apply to the spectrum.

        Returns
        -------
        np.ndarray
            The spectrum.
        """
        if self.array is None:
            return None
        
        spectrum = self.array[z, y, x, :]

        if phase:
            spectrum = spectrum * np.exp(-1j * phase)

        return spectrum
