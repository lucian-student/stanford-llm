from pydantic import BaseModel, Field
from typing import Dict, Union, List, Any, Tuple
from configuration_engine.parameter import (
    RangeParameterSchema,
    NontunableParameter,
    Parameter,
    ConstantNontunableParameter,
    BaseParameter,
    ConstantParameter,
)
from configuration_engine.configuration import Metadata
from cs336_basics.dataset import SequeunceDataset
import optuna

ParameterType = Union[
    RangeParameterSchema[int],
    RangeParameterSchema[float],
    int,
    float,
    str,
    bool,
    List[str],
]


class TrainingConfiguration:

    def __init__(
        self,
        metadata: Metadata,
        additional_parameters: List[NontunableParameter[Any]],
        tuner_parameters: List[NontunableParameter[Any]],
        training_dataset: SequeunceDataset,
        validation_dataset: SequeunceDataset,
        training_parameters: List[Parameter[Any]],
        model_parameters: List[Parameter[Any]],
        optimizer_parameters: List[Parameter[Any]],
    ):
        self.additional_parameters = additional_parameters
        self.tuner_parameters = tuner_parameters
        self.metadata = metadata
        self.training_dataset = training_dataset
        self.validation_datset = validation_dataset
        self.optimizer_parameters = optimizer_parameters
        self.training_parameters = training_parameters
        self.model_parameters = model_parameters

    def suggest_optimizer_params(
        self, trial: optuna.Trial
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        params: Dict[str, Any] = {}
        for param in self.optimizer_parameters:
            params[param.name] = param.suggest(trial)
        return params, params

    def suggest_model_params(
        self, trial: optuna.Trial
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        params: Dict[str, Any] = {}
        for param in self.model_parameters:
            params[param.name] = param.suggest(trial)
        return params, params

    def suggest_training_params(
        self, trial: optuna.Trial
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        params: Dict[str, Any] = {}
        for param in self.training_parameters:
            params[param.name] = param.suggest(trial)
        return params, params

    def construct_additional_params(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        params: Dict[str, Any] = {}
        for param in self.additional_parameters:
            params[param.name()] = param.value()
        return params, params

    def construct_tuner_parameters(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        params: Dict[str, Any] = {}
        for param in self.tuner_parameters:
            params[param.name()] = param.value()
        return params, params

    def first_model_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for param in self.model_parameters:
            params[param.name] = param.first()
        return params


class Dataset(BaseModel):
    file_path: str
    sequence_length: int


class TrainingSchema(BaseModel):
    metadata: Metadata
    train_dataset: Dataset
    validation_dataset: Dataset
    additional_parameters: Dict[str, Union[int, float, str, bool]]
    tuner_parameters: Dict[str, Union[int, float, str, bool]]
    training_parameters: Dict[str, ParameterType]
    model_parameters: Dict[str, ParameterType]
    optimizer_parameters: Dict[str, ParameterType]

    def convert_paramaeters(
        self,
        parameters: Dict[str, ParameterType],
    ) -> List[Parameter]:
        converted: List[Parameter] = []
        for key, val in parameters.items():
            if isinstance(val, BaseParameter):
                converted.append(val.build(key))
            else:
                converted.append(ConstantParameter(name=key, value=val))
        return converted

    def build(self) -> TrainingConfiguration:
        converted_optimizer_parameters: List[Parameter[Any]] = self.convert_paramaeters(
            self.optimizer_parameters
        )
        converted_training_parameters: List[Parameter[Any]] = self.convert_paramaeters(
            self.training_parameters
        )
        converted_model_parameters: List[Parameter[Any]] = self.convert_paramaeters(
            self.model_parameters
        )
        converted_tuner_parameters: List[NontunableParameter[Any]] = []
        for key, val in self.tuner_parameters.items():
            converted_tuner_parameters.append(
                ConstantNontunableParameter(name=key, value=val)
            )
        converted_additional_parameters: List[NontunableParameter[Any]] = []
        for key, val in self.additional_parameters.items():
            converted_additional_parameters.append(
                ConstantNontunableParameter(name=key, value=val)
            )
        train_dataset = SequeunceDataset(**self.train_dataset.model_dump())
        validation_datset = SequeunceDataset(**self.validation_dataset.model_dump())
        return TrainingConfiguration(
            metadata=self.metadata,
            additional_parameters=converted_additional_parameters,
            tuner_parameters=converted_tuner_parameters,
            training_dataset=train_dataset,
            validation_dataset=validation_datset,
            training_parameters=converted_training_parameters,
            model_parameters=converted_model_parameters,
            optimizer_parameters=converted_optimizer_parameters,
        )
