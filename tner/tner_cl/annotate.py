import argparse
import inspect
import json
import logging
import numpy as np
import os
import regex as re
import rich

from copy import deepcopy
from dataclasses import dataclass, field, MISSING
from pathlib import Path
from rich import print as rprint
from rich.panel import Panel
from typing import Any, Dict, List, Optional, TextIO, Union

from tner.util import tokenize_sentence


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


with open(Path(__file__).parents[2] / 'unified_label2id.json', 'r') as f:
    ID2LABEL = {v: k for k, vals in json.load(f).items() for v in vals}
LABELS = set(ID2LABEL.values())


@dataclass
class Annotation:
    type: str = field(default=MISSING)
    entity: List[str] = field(default_factory=MISSING)
    position: List[int] = field(default_factory=MISSING)
    partial: bool = field(default=False)
    _color: bool = field(default=True, init=False)

    def __post_init__(self):
        self.set_type(self.type)

    def __len__(self):
        return len(self.entity)

    @property
    def boundaries(self):
        return slice(self.position[0], self.position[-1] + 1)

    @property
    def text(self):
        return to_string(self.entity)

    def set_type(self, hint):
        self.type = self.guess_type(hint)

    def display_tags(self):
        color = "bold italic blue" if self.partial else "bold blue"
        fmt = f"[{color}]{{}}[/{color}]" if self._color else f"{{}}"
        return fmt.format(f"<{self.type}>"), fmt.format(f"</{self.type}>")

    def toggle_color(self, value: bool):
        assert type(value) is bool
        self._color = value

    def __str__(self):
        return " {1} {0} {2}".format(to_string(self.entity), *self.display_tags())

    def to_json(self):
        obj = {
            "type": self.type,
            "entity": self.entity,
            "position": self.position,
        }
        if self.partial:
            obj["partial"] = self.partial
        return obj

    @staticmethod
    def guess_type(type_hint):
        transforms = [
            lambda x: x,
            lambda x: x.lower(),
            lambda x: x.upper(),
            lambda x: x.replace(" ", "_")
        ]
        for transform in transforms:
            if transform(type_hint) in LABELS:
                return transform(type_hint)
            elif transform(type_hint) in ID2LABEL:
                return ID2LABEL[transform(type_hint)]
        raise ValueError(f"Unknown type: {type_hint}")

@dataclass
class EntityPrediction(Annotation):
    probability: List[float] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        assert len(self.probability) == len(self.entity), "Probability and entity length mismatch"
        self.probability = np.array(self.probability)

    def display_tags(self):
        # fmt = f"{Fore.BLUE}{{}}{Style.RESET_ALL}"
        fmt = "[bold blue]{}[/bold blue]"
        P = np.mean(self.probability)
        return fmt.format(f"<{self.type} P={P:.2%}>"), fmt.format(f"</{self.type}>")

    def to_json(self):
        return {
            "type": self.type,
            "entity": self.entity,
            "position": self.position,
            "probability": self.probability.tolist(),
        }


@dataclass
class Example:
    id: str
    input: List[str]
    entity_prediction: Optional[List[EntityPrediction]] = field(default_factory=list)
    prediction: Optional[List[str]] = field(default_factory=list)
    probability: Optional[List[float]] = field(default=None)
    nll: float = field(default=None)
    annotation: Optional[List[Annotation]] = field(default_factory=list)
    translation: Optional[Dict[str,Any]] = field(default_factory=dict)
    _is_annotated: bool = field(default=False, init=False)
    is_translation: bool = field(default=False)

    def __post_init__(self):
        self.entity_prediction = [EntityPrediction(**kwargs) for kwargs in self.entity_prediction]
        self.annotation = [Annotation(**kwargs) for kwargs in self.annotation]
        self.translation = Example(**self.translation, is_translation=True) if self.translation else None

    @property
    def text(self):
        return to_string(self.input)

    def toggle_color(self, value:bool):
        for annot in self.annotation:
            annot.toggle_color(value)

    def copy(self):
        return deepcopy(self)

    def annotate(self, example_id:int, total_examples:int, output_file:Optional[TextIO]=None):
        while True:
            try:
                self.display(example_id, total_examples, preds=not(self._is_annotated))
                if not self._is_annotated:
                    # First time annotating: we initialize the annotations from the predictions
                    self.init_annotations()
                exit_code = self.run_command(input("Enter command: ").strip())
                if exit_code == 1:
                    logging.info(f"Example {self.id} annotated")
                    if output_file is not None:
                        self.save(output_file)
                if exit_code != 0:
                    return exit_code
            except Exception as e:
                if isinstance(e, (ValueError, TypeError, KeyError, IndexError)):
                    rprint(f"[red]Error: {e!r}[/red] ")
                    raise e
                    input("Press enter to continue...")
                    continue
                raise e

    def run_command(self, command):
        """
        Options:
            h: help
            q: quit
            a: add annotation
            d: delete annotation
            u: update annotation
            m: merge annotations
            n: go to next example without saving
            ↵: save

        Syntax:
            h labels
                list allowed entity types
            a [ID] TYPE ENTITY
                add annotation of TYPE to IDth ENTITY
                e.g. a 1 PER John Doe
            d ENTITY_ID [ENTITY_ID ...]
                delete ENTITY_IDth ENTITY
                e.g. d 1 2 3
            u ENTITY_ID [TYPE [ENTITY]] | ENTITY_ID [+-SHIFT [r]] | ENTITY_ID +p
                update ENTITY_IDth ENTITY or shift start/end position by SHIFT or toggle partial flag
                e.g. u 1 PER John Doe or u 1 +1 r
            m ENTITY_ID ENTITY_ID
                merge entities
                e.g. m 1 2
        """
        if command == '':
            return 1
        cmd, *remainder = command.split(maxsplit=1)
        if cmd not in ("q", "h", "debug", "p", "m", "n", "a", "d", "u"):
            rprint(f'Unknown command: {command}')
            return 0

        if cmd == 'q':
            return -1

        if cmd == 'h':
            if remainder and remainder[0] == "labels":
                rprint("\n".join(sorted(LABELS)))
                return 0
            rprint(inspect.getdoc(self.run_command))
            input("Press enter to continue...")
            return 0

        if cmd == 'debug':
            import ipdb; ipdb.set_trace()
            return 0

        if cmd == 'p':
            rprint(to_string(self.input))
            input("Press enter to continue...")
            return 0

        if cmd == 'n':
            return 2

        assert remainder != [], {
            "a": "a [ID] TYPE ENTITY",
            "d": "d ENTITY_ID [ENTITY_ID ...]",
            "m": "m ENTITY_ID ENTITY_ID",
            "u": "u ENTITY_ID [TYPE [ENTITY]] | ENTITY_ID [+-SHIFT [r]] | ENTITY_ID +p",
        }[cmd]

        if cmd == 'a':
            typ, entity = remainder[0].split(maxsplit=1)
            _id = 0
            if typ.isdigit():
                _id = int(typ) - 1
                typ, entity = entity.split(maxsplit=1)
            positions, tokens = self.find_positions(entity, _id)
            self.add_annotation(typ, tokens, positions)
            return 0

        if cmd == 'd':
            entity_ids = list(map(int, remainder[0].split()))
            for _id in reversed(sorted(entity_ids)):
                self.delete_annotation(_id)
            return 0

        if cmd == 'u':
            kwargs = {}
            index, remainder = remainder[0].split(maxsplit=1)
            indices = self._parse_index(index)
            if re.match("[\+\-]\d+( r)?", remainder):
                kwargs["shift"], *remainder = remainder.split(maxsplit=1)
                kwargs["reverse"] = bool(remainder and remainder[0] == "r")
            elif re.match("[\+\-]p", remainder):
                kwargs["toggle_partial"] = True
            else:
                typ, *remainder = remainder.split(maxsplit=1)
                kwargs["typ"] = typ
                if remainder:
                    kwargs["entity"] = remainder[0]
            for index in indices:
                kwargs["index"] = index
                self.update_annotation(**kwargs)
            return 0

        if cmd == 'm':
            i, j = list(map(int, remainder[0].split()))
            self.merge_annotations(i, j)
            return 0

    @staticmethod
    def _parse_index(index_string):
        assert re.match(r"(\d(\,)?)+", index_string)
        if not index_string.isdigit():
            return tuple(int(i) - 1 for i in index_string.split(","))
        return (int(index_string) - 1,)

    @property
    def has_translation(self):
        return self.translation is not None

    def init_annotations(self):
        if self._is_annotated:
            raise ValueError("Example already annotated")
        self.annotation = [Annotation(e.type, e.entity, e.position) for e in self.entity_prediction]
        self._is_annotated = True

    def add_annotation(self, typ, entity, position, partial=False):
        self.annotation.append(Annotation(typ, entity, position, partial))
        self.annotation.sort(key=lambda x: x.position[0])
        self.ensure_no_overlap()

    def delete_annotation(self, i):
        assert i in range(1, len(self.annotation) + 1)
        # rprint("Delete annotation: ", self.annotation[i - 1]); input("Press enter to continue...")
        del self.annotation[i - 1]

    def ensure_no_overlap(self, remove=False):
        to_remove = []
        for i, annotation in enumerate(self.annotation):
            if i == 0:
                continue
            prev_annotation = self.annotation[i-1]
            if prev_annotation.position[-1] > annotation.position[0]:
                if remove:
                    to_remove.append(i)
                else:
                    raise ValueError(f"Overlapping annotations: {prev_annotation} and {annotation}")
        for i in reversed(to_remove):
            logging.warning(f"Removing overlapping annotation: {self.annotation[i]}")
            del self.annotation[i]

    def update_annotation(self, index, typ=None, entity=None, shift:str=None, reverse=True, toggle_partial=False):
        annot = self.annotation[index]
        typ = typ if typ is not None else annot.type
        partial = not(annot.partial) if toggle_partial else annot.partial
        if shift:
            if re.match(r"[\-\+]\d+", shift):
                min_bound, max_bound = annot.position[0], annot.position[-1]
                shift = int(shift[1:]) * (1 if (shift[0] == "+") ^ reverse else -1)
                if reverse:
                    min_bound = annot.position[0] + shift
                else:
                    max_bound = annot.position[-1] + shift
                if not(0 <= min_bound <= max_bound < len(self.input)):
                    raise ValueError(f"Shift {shift} out of bounds for entity {index}: {annot.entity}")
                positions = list(range(min_bound, max_bound + 1))
                entity = [self.input[pos] for pos in positions]
            else:
                raise ValueError(f"Invalid shift value: {shift}")
        else:
            if entity is not None:
                positions, entity = self.find_positions(entity)
            else:
                positions, entity = annot.position, annot.entity
        self.annotation.pop(index)
        self.add_annotation(typ, entity, positions, partial)

    def find_positions(self, entity, id=0):
        tokens = self.input
        _id = -1
        tokenized_entity = tokenize_sentence(" " + entity.strip())
        for i in range(len(tokens)):
            candidate = ''.join(tokens[i:i+len(tokenized_entity)]).strip()
            if candidate == entity:
                _id += 1
                if _id == id:
                    return list(range(i, i + len(tokenized_entity))), tokenized_entity
        raise ValueError(f"Entity {entity} not found in sentence")

    def merge_annotations(self, i, j):
        if i > j:
            i, j = j, i
        annot2 = self.annotation.pop(j-1)
        annot1 = self.annotation.pop(i-1)
        if annot1.position[0] > annot2.position[0]:
            annot1, annot2 = annot2, annot1
        min_pos, max_pos = annot1.position[0], annot2.position[-1]
        positions = list(range(min_pos, max_pos + 1))
        entity = [self.input[pos] for pos in positions]
        self.add_annotation(annot1.type, entity, positions)

    def to_json(self, pred=True):
        data = {
            "id": self.id,
            "input": self.input,
            "annotation": [annot.to_json() for annot in self.annotation],
        }
        if pred:
            data.update({
                "entity_prediction": [entity.to_json() for entity in self.entity_prediction],
                "prediction": self.prediction,
                "probability": self.probability,
                "nll": self.nll
            })
        return data

    def save(self, output_file:Union[Path,str,TextIO], append=False, **kwargs):
        if isinstance(output_file, (Path, str)):
            flag = "a" if Path(output_file).exists() and append else "w"
            with open(output_file, flag) as f:
                self.save(f)
            return
        print(json.dumps(self.to_json(), **kwargs), file=output_file)
        output_file.flush()

    def __str__(self):
        annots = deepcopy(self.annotation)
        annots = iter(enumerate(annots, 1))
        annot, next_annot = None, next(annots, None)
        text = ""
        for i in range(len(self.input)):
            if next_annot and i == next_annot[1].position[0]:
                text += f"{next_annot[1]} "
                annot, next_annot = next_annot, next(annots, None)
            elif annot and i in annot[1].position:
                continue
            else:
                text += self.input[i]
        return text

    def display(self, example_id:Optional[int]=None, total_examples:Optional[int]=None, preds=False, selected_ids="all", clear_screen=True):
        if clear_screen:
            clear()
        if not self.is_translation:
            title = f"{self.id}" \
                + (f" {example_id}/{total_examples}" if example_id is not None and total_examples is not None else "") \
                + (f" [red][{self.nll:.3f}][/red]" if self.nll else "")
        elif self.is_translation:
            title = "Translation"

        annots = deepcopy(
            self.annotation if not preds
            else self.entity_prediction
        )

        if selected_ids:
            if selected_ids == "all":
                selected_ids = list(range(1, len(annots) + 1))

            if isinstance(selected_ids, int):
                selected_ids = [selected_ids]

            assert type(selected_ids is list)
            assert all(type(id) is int for id in selected_ids)

        annots = iter(enumerate(annots, 1))
        annot, next_annot = None, next(annots, None)
        text = ""
        for i in range(len(self.input)):
            if next_annot and i == next_annot[1].position[0]:
                if next_annot[0] not in selected_ids:
                    text += f" {next_annot[1].text}"
                else:
                    text += f"{next_annot[1]} "
                annot, next_annot = next_annot, next(annots, None)
            elif annot and i in annot[1].position:
                continue
            else:
                text += self.input[i]
        rprint(Panel(text, title=title))
        rprint("\n")

        self.display_annotations(preds=preds, selected_ids=selected_ids)
        if self.has_translation:
            self.translation.display(clear_screen=False, preds=True)

    def display_annotations(self, preds=False, selected_ids=None):
        annotations = self.entity_prediction if preds else self.annotation
        selected_ids = selected_ids or []
        for i, annot in enumerate(annotations, 1):
            if i not in selected_ids:
                continue
            rprint(f"\t[green][{i}] [/green]{annot} [{annot.position[0]}:{annot.position[-1]}]")
        rprint("\n")


class Annotator:

    @classmethod
    def from_args(cls, args=None):
        parser = argparse.ArgumentParser(description='command line tool to annotate NER corpus',)
        parser.add_argument('-f', '--file', help='json lines file containing the predictions to correct', required=True, type=str)
        parser.add_argument('-o', '--output', help='output file in json lines format', required=True, type=str)
        parser.add_argument('-s', '--sort', help='sort the examples by confidence score', action="store_true")
        parser.add_argument('-t', '--translation_file', help='file containing the annotated translations', default=None, type=str)
        parser.add_argument('-w', '--width', help='width of the terminal', default=100, type=int)
        return cls(parser.parse_args(args))

    def __init__(self, options):
        self.output_file = Path(options.output)
        self.input_file = Path(options.file)
        self.sort = options.sort
        self.translation_file = options.translation_file
        if options.width < os.get_terminal_size()[0]:
            rich.reconfigure(width=options.width)

        self.append = False
        if self.output_file.exists():
            self.append = True
            logging.info(f'Appending to {self.output_file}')

    def get_annotated_ids(self):
        if not self.append:
            return set()

        with open(self.output_file, 'r') as output_file:
            return {json.loads(line)["id"] for line in output_file.readlines()}

    def run(self):
        discard_ids = self.get_annotated_ids()
        done = len(discard_ids)
        with open(self.input_file, 'r') as input_file:
            results = [json.loads(line) for line in input_file.readlines()]
            total = len(results)
            if self.append and discard_ids:
                results = [result for result in results if not(result['id'] in discard_ids)]
            if self.sort:
                results = sorted(results, key=lambda x: x['nll'], reverse=True)

        num_translations = 0
        if self.translation_file:
            with open(self.translation_file, 'r') as translation_file:
                translations = [json.loads(line) for line in translation_file.readlines()]
                translations = {translation['id']: translation for translation in translations}
                for result in results:
                    if result['id'] in translations:
                        num_translations += 1
                        result['translation'] = translations[result['id']]

        rprint(f"Found {total:,} examples with {num_translations} translations."
              f"{len(discard_ids):,} annotated, {len(results):,} remaining")
        rprint("Press Ctrl+C to exit\n")

        with open(self.output_file, 'a' if self.append else 'w') as output_file:
            for i, example in enumerate(results, done + 1):
                example = Example(**example)
                exit_code = example.annotate(i, total, output_file)
                if exit_code == -1:
                    rprint("Exiting...")
                    return

        rprint("\n")
        rprint(f"Finished annotating {i - done:,} examples ({done:,} already annotated)")


def to_string(obj:List[str]):
    return "".join(obj).strip()


def clear():
    # os.system("clear")
    rich.get_console().clear()


def check_for_keyboard_interrupt(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            return False
    return wrapper


def debug(func):
    import ipdb
    def wrapper(*args, **kwargs):
        with ipdb.launch_ipdb_on_exception():
            return func(*args, **kwargs)
    return wrapper


@debug
@check_for_keyboard_interrupt
def main():
    annotator = Annotator.from_args()
    annotator.run()
