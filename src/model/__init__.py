from src.model.segresnet import SegResNetDetector
from src.model.swinunetr import SwinUNETRDetector
from src.model.dict_sequential import DictSequential
from src.model.sliding_window_wrapper import SlidingWindowWrapper
from src.model.compile_wrapper import CompileWrapper
from src.model.nnunet import nnUNetDetector, SSLnnUNetDetector
from src.model.ensemble import Ensemble
from src.model.ssl_wrapper import NNUnetMAEStructSemantic
