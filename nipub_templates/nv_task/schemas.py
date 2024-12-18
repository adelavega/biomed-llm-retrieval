from typing import List
from pydantic import BaseModel, Field


class TaskMetadataModel(BaseModel):
    # Annotated fields
    Modality: List[str] = Field(
        description="Modality", 
        enum=["fMRI-BOLD", "Structural MRI", "Diffusion MRI", "PET FDG", "PET [15O]-water", "fMRI-CBF", "fMRI-CBV", "MEG", "EEG", "Other"],
    )
    DesignType: List[str] = Field(
        description="Design Type", enum=["Task-based", "RestingState"])
    Exclude: str = Field(
        description="Exclude", enum=["Meta-Analysis", "Review"]
        )
    TaskName: List[str] = Field(
        description="Task (or lack of it) performed by the subjects in the scanner"
        )
    TaskDescription: List[str] = Field(
        description="Text describing in detail the task performed by the subject(s) in the scanner. If multiple tasks are performed, provide a description for each task."
        )
    Condition: List[str] = Field(
        description="Conditions of task performed by the subjects in the scanner"
        )
    ContrastDefinition: List[str] = Field(
        description="Terms or conditions that are compared (and results are reported) in the fMRI task (e.g. 'faces vs. houses', '1-back > 0-back')."
        )


class NeuroVaultDetailedMetdata(BaseModel):
    TypeOfDesign: str = Field(
        description="Blocked, event-related, hybrid block/event, or other",
        enum=["blocked", "eventrelated", "hybridblockevent", "other"]
    )
    NumberOfImagingRuns: int = Field(
        description="Number of imaging runs acquired"
    )
    NumberOfExperimentalUnits: int = Field(
        description="Number of blocks, trials or experimental units per imaging run"
    )
    LengthOfRuns: float = Field(
        description="Length of each imaging run in seconds"
    )
    LengthOfBlocks: float = Field(
        description="For blocked designs, length of blocks in seconds"
    )
    LengthOfTrials: str = Field(
        description="Length of individual trials in seconds. If length varies across trials, enter 'variable'. "
    )
    Optimization: bool = Field(
        description="Was the design optimized for efficiency"
    )
    OptimizationMethod: str = Field(
        description="What method was used for optimization?"
    )
    SubjectAgeMean: float = Field(
        description="Mean age of subjects"
    )
    SubjectAgeMin: float = Field(
        description="Minimum age of subjects"
    )
    SubjectAgeMax: float = Field(
        description="Maximum age of subjects"
    )
    Handedness: str = Field(
        description="Handedness of subjects",
        enum=["right", "left", "both"]
    )
    ProportionMaleSubjects: float = Field(
        description="The proportion (not percentage) of subjects"
    )
    InclusionExclusionCriteria: str = Field(
        description="Additional inclusion/exclusion criteria, if any (including specific sampling strategies that limit inclusion to a specific group, such as laboratory members)"
    )
    NumberOfRejectedSubjects: int = Field(
        description="Number of subjects scanned but rejected from analysis"
    )
    GroupComparison: bool = Field(
        description="Was this study a comparison between subject groups?"
    )
    GroupDescription: str = Field(
        description="A description of the groups being compared"
    )
    ScannerMake: str = Field(
        description="Manufacturer of MRI scanner"
    )
    ScannerModel: str = Field(
        description="Model of MRI scanner"
    )
    FieldStrength: float = Field(
        description="Field strength of MRI scanner (in Tesla)"
    )
    PulseSequence: str = Field(
        description="Description of pulse sequence used for fMRI"
    )
    ParallelImaging: str = Field(
        description="Description of parallel imaging method and parameters"
    )
    FieldOfView: float = Field(
        description="Imaging field of view in millimeters"
    )
    MatrixSize: int = Field(
        description="Matrix size for MRI acquisition"
    )
    SliceThickness: float = Field(
        description="Distance between slices (includes skip or distance factor) in millimeters"
    )
    SkipDistance: float = Field(
        description="The size of the skipped area between slices in millimeters"
    )
    AcquisitionOrientation: str = Field(
        description="The orientation of slices"
    )
    OrderOfAcquisition: str = Field(
        description="Order of acquisition of slices (ascending, descending, or interleaved)",
        enum=["ascending", "descending", "interleaved"]
    )
    RepetitionTime: float = Field(
        description="Repetition time (TR) in milliseconds"
    )
    EchoTime: float = Field(
        description="Echo time (TE) in milliseconds"
    )
    FlipAngle: float = Field(
        description="Flip angle in degrees"
    )
    SoftwarePackage: str = Field(
        description="If a single software package was used for all analyses, specify that here"
    )
    SoftwareVersion: str = Field(
        description="Version of software package used"
    )
    OrderOfPreprocessingOperations: str = Field(
        description="Specify order of preprocessing operations"
    )
    QualityControl: str = Field(
        description="Describe quality control measures"
    )
    UsedB0Unwarping: bool = Field(
        description="Was B0 distortion correction used?"
    )
    B0UnwarpingSoftware: str = Field(
        description="Specify software used for distortion correction if different from the main package"
    )
    UsedSliceTimingCorrection: bool = Field(
        description="Was slice timing correction used?"
    )
    SliceTimingCorrectionSoftware: str = Field(
        description="Specify software used for slice timing correction if different from the main package"
    )
    UsedMotionCorrection: bool = Field(
        description="Was motion correction used?"
    )
    MotionCorrectionSoftware: str = Field(
        description="Specify software used for motion correction if different from the main package"
    )
    MotionCorrectionReference: str = Field(
        description="Reference scan used for motion correction"
    )
    MotionCorrectionMetric: str = Field(
        description="Similarity metric used for motion correction"
    )
    MotionCorrectionInterpolation: str = Field(
        description="Interpolation method used for motion correction"
    )
    UsedMotionSusceptibiityCorrection: bool = Field(
        description="Was motion-susceptibility correction used?"
    )
    UsedIntersubjectRegistration: bool = Field(
        description="Were subjects registered to a common stereotactic space?"
    )
    IntersubjectRegistrationSoftware: str = Field(
        description="Specify software used for intersubject registration if different from main package"
    )
    IntersubjectTransformationType: str = Field(
        description="Was linear or nonlinear registration used?",
        enum=["linear", "nonlinear"]
    )
    NonlinearTransformType: str = Field(
        description="If nonlinear registration was used, describe transform method"
    )
    TransformSimilarityMetric: str = Field(
        description="Similarity metric used for intersubject registration"
    )
    InterpolationMethod: str = Field(
        description="Interpolation method used for intersubject registration"
    )
    ObjectImageType: str = Field(
        description="What type of image was used to determine the transformation to the atlas? (e.g. T1, T2, EPI)"
    )
    FunctionalCoregisteredToStructural: bool = Field(
        description="Were the functional images coregistered to the subject's structural image?"
    )
    FunctionalCoregistrationMethod: str = Field(
        description="Method used to coregister functional to structural images"
    )
    CoordinateSpace: str = Field(
        description="Name of coordinate space for registration target",
        enum=["mni", "talairach", "mni2tal", "other"]
    )
    TargetTemplateImage: str = Field(
        description="Name of target template image"
    )
    TargetResolution: float = Field(
        description="Voxel size of target template in millimeters"
    )
    UsedSmoothing: bool = Field(
        description="Was spatial smoothing applied?"
    )
    SmoothingType: str = Field(
        description="Describe the type of smoothing applied"
    )
    SmoothingFWHM: float = Field(
        description="The full-width at half-maximum of the smoothing kernel in millimeters"
    )
    ResampledVoxelSize: float = Field(
        description="Voxel size in mm of the resampled, atlas-space images"
    )
    IntrasubjectModelType: str = Field(
        description="Type of model used (e.g., regression)"
    )
    IntrasubjectEstimationType: str = Field(
        description="Estimation method used for model (e.g., OLS, generalized least squares)"
    )
    IntrasubjectModelingSoftware: str = Field(
        description="Software used for intrasubject modeling if different from overall package"
    )
    HemodynamicResponseFunction: str = Field(
        description="Nature of HRF model"
    )
    UsedTemporalDerivatives: bool = Field(
        description="Were temporal derivatives included?"
    )
    UsedDispersionDerivatives: bool = Field(
        description="Were dispersion derivatives included?"
    )
    UsedMotionRegressors: bool = Field(
        description="Were motion regressors included?"
    )
    UsedReactionTimeRegressor: bool = Field(
        description="Was a reaction time regressor included?"
    )
    UsedOrthogonalization: bool = Field(
        description="Were any regressors specifically orthogonalized with respect to others?"
    )
    OrthogonalizationDescription: str = Field(
        description="If orthogonalization was used, describe here"
    )
    UsedHighPassFilter: bool = Field(
        description="Was high pass filtering applied?"
    )
    HighPassFilterMethod: str = Field(
        description="Describe method used for high pass filtering"
    )
    AutocorrelationModel: str = Field(
        description="What autocorrelation model was used (or 'none' of none was used)"
    )
    GroupModelType: str = Field(
        description="Type of group model used (e.g., regression)"
    )
    GroupEstimationType: str = Field(
        description="Estimation method used for group model (e.g., OLS, generalized least squares)"
    )
    GroupModelingSoftware: str = Field(
        description="Software used for group modeling if different from overall package"
    )
    GroupInferenceType: str = Field(
        description="Type of inference for group model",
        enum=["randommixedeffects", "fixedeffects"]
    )
    GroupModelMultilevel: str = Field(
        description="If more than 2-levels, describe the levels and assumptions of the model (e.g. are variances assumed equal between groups)"
    )
    GroupRepeatedMeasures: bool = Field(
        description="Was this a repeated measures design at the group level?"
    )
    GroupRepeatedMeasuresMethod: str = Field(
        description="If multiple measurements per subject, list method to account for within subject correlation, exact assumptions made about correlation/variance"
    )
