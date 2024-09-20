import itk
import numpy as np
import skimage
from skimage.morphology import isotropic_closing
from typing_extensions import Self

from Onix.utils import register_resample, align_map


class OnixObject:
    """
    Base class for Onix objects.
    """
    def __init__(self):
        self._silent = True

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __str__(self):
        return f"{self.__class__.__name__}"

    def _silence(self):
        self._silent = True

    def _unsilence(self):
        self._silent = False


class OnixVolume(OnixObject):
    """
    Class that represents a multi-dimensional image.

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

    def __init__(self, array: np.ndarray, image: itk.Image, is_mask: bool = False):
        super().__init__()
        self.array = array
        self.image = image
        self.is_mask = is_mask

    def register(self, fixed: Self, tx) -> Self:
        """
        Register itk data to the fixed volume using the specified transform.

        Parameters
        ----------
        fixed : OnixVolume
            The fixed image to register to.
        tx : itk.Transform
            The transform to use for registration.
        """
        image = register_resample(fixed.image, self.image, tx)
        return OnixVolume(self.array, image, self.is_mask)

    def align(self, fixed: Self) -> Self:
        """
        Align itk data to the coordinate system of the fixed volume.

        Parameters
        ----------
        fixed : OnixVolume
            The fixed image to align to.

        Returns
        -------
        OnixVolume
            Volume with aligned image.
        """
        image = align_map(self.image, fixed.image)
        return OnixVolume(self.array, image, self.is_mask)

    def connected_component(self) -> Self:
        """
        Find the largest connected component in the mask.

        Returns
        -------
        OnixVolume
            The largest connected component.
        """
        if self.is_mask:
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
            image = itk.GetImageFromArray(array)
            image.SetOrigin(self.image.GetOrigin())
            image.SetSpacing(self.image.GetSpacing())
            image.SetDirection(self.image.GetDirection())
            return OnixVolume(array, image, is_mask=True)
        else:
            if self._silent:
                return None
            else:
                raise Exception(f"Cannot apply connected component to non-mask")

    def slicer_array(self) -> np.ndarray:
        """
        Get the image data as a numpy array for the slicer.
        """
        return slicer_array(self.image)

    def flip_x(self) -> Self:
        """
        Flip the volume along the x-axis.
        """
        array = self.array[:, :, ::-1]
        image = self.image
        flipFilter = itk.FlipImageFilter[image].New()
        flipFilter.SetInput(image)
        flipAxes = (True, False, False)
        flipFilter.SetFlipAxes(flipAxes)
        flipFilter.Update()
        image = flipFilter.GetOutput()
        return OnixVolume(array, image, self.is_mask)


class OnixSpectra(OnixObject):
    """
    Represents a 4D spectroscopic image.

    Assumes the data is in the shape (z, y, x, f).

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

    def flip_x(self) -> Self:
        """
        Flip the data along the x-axis.
        """
        array = self.array[:, :, ::-1, :]
        return OnixSpectra(array)

    def get(
        self, 
        x: int, 
        y: int, 
        z: int, 
        component: str = "real", 
        phase: float = 0.,
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
            # TODO work with arbitrary shapes
            spectrum = np.zeros(512)
        else:
            spectrum = self.array[z, y, x, :]

        if phase:
            spectrum = spectrum * np.exp(-1j * phase)

        if component == "real":
            return spectrum.real
        elif component == "imaginary":
            return spectrum.imag
        elif component == "magnitude":
            return np.abs(spectrum)
        else:
            raise Exception("Invalid component")
