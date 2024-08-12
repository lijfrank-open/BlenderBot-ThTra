from inputters.strat import Inputter as strat
from inputters.vanilla import Inputter as vanilla
from inputters.strat_with_predict import Inputter as strat_with_predict

inputters = {
    'vanilla': vanilla,
    'strat': strat,
    'strat_with_predict': strat_with_predict,

}
