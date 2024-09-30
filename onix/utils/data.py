import SimpleITK as sitk
import numpy as np


def rigid_registration(
    fixed: sitk.Image,
    moving: sitk.Image,
    learn_rate: float = 0.01,
    stop: float = 0.001,
    max_steps: int = 50,
    log: bool = False,
) -> sitk.Image, sitk.Transform:
    """
    Register two images using the specified parameters.

    Uses a VersorRigid3DTransform as the initial transform.
    Uses a regular step gradient descent optimizer with mean squares metric.

    Parameters
    ----------
    fixed : sitk.Image
        The fixed image to register to.
    moving : sitk.Image
        The moving image to register.
    learn_rate : float, optional
        The learning rate for the optimizer, by default 4.0
    stop : float, optional
        The stopping criteria for the optimizer, by default 0.01
    max_steps : int, optional
        The maximum number of steps for the optimizer, by default 200
    log : bool, optional
        Whether to log the optimizer progress, by default False

    Returns
    -------
    sitk.Image
        The registered image.
    sitk.Transform
        The transform used for registration.
    """
    def command_iteration(method):
        print(
            f"{method.GetOptimizerIteration():3} = "
            f"{method.GetMetricValue():10.5f} : "
            f"{method.GetOptimizerPosition()}"
        )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(learn_rate, stop, max_steps)
    R.SetInitialTransform(sitk.VersorRigid3DTransform())
    R.SetInterpolator(sitk.sitkLinear)

    if log:
        R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    tx = R.Execute(fixed, moving)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)

    out = resampler.Execute(moving)

    return out, tx

def mask_uint8_map(self, image: sitk.Image, mask: sitk.Image):
    """
    Mask the sitkUInt8 image using the binary mask.

    Parameters
    ----------
    image : sitk.Image
        The image to mask. Must be of type uint8.
    mask : sitk.Image
        The binary mask to use for masking. Must be of type uint8.

    Returns
    -------
    sitk.Image
        The masked image.
    """
    assert image.GetPixelID() == sitk.sitkUInt8
    assert binary_mask.GetPixelID() == sitk.sitkUInt8
    B = sitk.MaskImageFilter()
    return B.Execute(image, mask_image)

def binary_threshold(self, image: sitk.Image, lower: float = 1., upper: float = 255.):
    """
    Apply a binary threshold to the image.

    Parameters
    ----------
    image : sitk.Image
        The image to threshold.
    lower : float, optional
        The lower threshold, by default 1.
    upper : float, optional
        The upper threshold, by default 255.
    """
    T = sitk.BinaryThresholdImageFilter()
    T.SetLowerThreshold(lower)
    T.SetUpperThreshold(upper)
    T.SetOutsideValue(0)
    T.SetInsideValue(1)
    return T.Execute(image)

def cast_uint8(self, image: sitk.Image):
    """
    Cast the image to sitkUInt8.

    Parameters
    ----------
    image : sitk.Image
        The image to cast.

    Returns
    -------
    sitk.Image
        The casted image
    """
    C = sitk.CastImageFilter()
    C.SetOutputPixelType(sitk.sitkUInt8)
    return C.Execute(image)

def threshold_above(image, threshold):
    """
    Threshold the image above the specified value.

    Parameters
    ----------
    image : sitk.Image
        The image to threshold.
    threshold : float
        The threshold value.

    Returns
    -------
    sitk.Image
        The thresholded image.
    """
    F = sitk.ThresholdImageFilter()
    F.ThresholdAbove(threshold)
    F.SetOutsideValue(threshold)
    return F.Execute(image)

def threshold_below(image, threshold):
    """ 
    Threshold the image below the specified value.

    Parameters
    ----------
    image : sitk.Image
        The image to threshold.
    threshold : float
        The threshold value.

    Returns
    -------
    sitk.Image
        The thresholded image.
    """
    F = sitk.ThresholdImageFilter()
    F.ThresholdBelow(threshold)
    F.SetOutsideValue(threshold)
    return F.Execute(image)

def align_image(image_1: sitk.Image, image_2: sitk.Image):
    """ 
    Align the first image to the second image's coordinate system.

    Parameters
    ----------
    """
    identity = sitk.Transform(3, sitk.sitkIdentity)
    R = sitk.ResampleImageFilter()
    R.SetTransform(identity)
    R.SetReferenceImage(image_2)
    return R.Execute(image_1)

def register_resample(fixed: sitk.Image, moving: sitk.Image, tx: sitk.Transform):
    """
    Register and resample the moving image to the fixed image.

    Parameters
    ----------
    fixed : sitk.Image
        The fixed image to register to.
    moving : sitk.Image
        The moving image to register.
    tx : sitk.Transform
        The transform to use for registration.

    Returns
    -------
    sitk.Image
        The resampled image.
    """
    R = sitk.ResampleImageFilter()
    R.SetReferenceImage(fixed)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetDefaultPixelValue(0)
    R.SetTransform(tx)
    return R.Execute(moving)

def cross_product(xr, yr, zr, xc, yc, zc):
    """ 
    Compute the cross product of two vectors.
    """
    return [yr * zc - zr * yc, zr * xc - xr * zc, xr * yc - yr * xc]

def update_origin(x, y, z, xr, yr, zr, xc, yc, zc):
    """ 
    Update the origin using the specified parameters.

    Parameters
    ----------
    x : float
        The x-coord of the origin.
    y : float
        The y-coord of the origin.
    z : float
        The z-coord of the origin.
    xr : float
        The x-coord of the 1st coordinate vector.
    yr : float
        The y-coord of the 1st coordinate vector.
    zr : float
        The z-coord of the 1st coordinate vector.
    xc : float
        The x-coord of the 2nd coordinate vector.
    yc : float
        The y-coord of the 2nd coordinate vector.
    zc : float
        The z-coord of the 2nd coordinate vector.
    """
    DCM = np.array(
        [
            [
                xr,
                yr,
                zr,
            ],
            [
                xc,
                yc,
                zc,
            ],
            cross_product(xr, yr, zr, xc, yc, zc),
        ],
        dtype=np.float64,
    )

    origin = np.array(
        [
            [x],
            [y],
            [z],
        ],
        dtype=np.float64,
    )

    new_origin = DCM @ origin

    return list(new_origin.reshape(-1))

def slicer_array(image: sitk.Image) -> np.ndarray:
    """
    Get the image data as a flipped numpy array for the slicer.

    Parameters
    ----------
    image : sitk.Image
        The image to get the data from.

    Returns
    -------
    np.ndarray
        The image data as a numpy array.
    """
    return np.flipud(sitk.GetArrayFromImage(image))
