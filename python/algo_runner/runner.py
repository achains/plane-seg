from pathlib import Path
from shutil import rmtree
from utils.cloud_builder import labels_to_cloud

import open3d as o3d
import os
import subprocess


__all__ = ["Runner"]


class RunnerException(BaseException):
    def __init__(self, msg: str):
        self.msg = msg


class Config:
    def __init__(self) -> None:
        self.config_path = None
        self.parameters = dict()

    @classmethod
    def from_file(cls, config_path: Path, delim="="):
        cfg = cls()
        cfg.config_path = config_path
        cfg.parameters = Config.parse_config(config_path, delim)

        return cfg

    @staticmethod
    def parse_config(config_path: Path, delim="="):
        parameters = None
        with open(config_path) as param_list:
            parameters = dict()
            for param_line in param_list:
                p_name, value = param_line.strip().split(delim)
                parameters.update({p_name: value})

        return parameters


class Runner:
    def __init__(self) -> None:
        self.executable_path = None
        self.labeled_pc_path = None
        self.pc_path = None
        self.config = None
        self.input_folder = Path("input")
        self.output_folder = Path("output")
        self.pc_name = Path("result.ply")
        self.labels_name = Path("planes.txt")
        self.prompt = ">>> "
        self.commands = (
            "load_pc",
            "set_labeled_path",
            "load_config",
            "load_executable",
            "run_alg",
            "init",
        )
        self.__runner_cfg = Path(__file__).parent / Path("runner.config")

    def do_load_pc(self, pc_path: str):
        pc = o3d.io.read_point_cloud(pc_path)
        pc.paint_uniform_color([0, 0, 0])

        if not os.path.exists(self.input_folder):
            os.mkdir(self.input_folder)

        self.pc_path = self.input_folder / self.pc_name
        o3d.io.write_point_cloud(str(self.pc_path), pc, write_ascii=True)
        print(f"Saved preprocessed pointcloud to {self.pc_path}")
        return 0

    def do_set_labeled_path(self, labeled_pc_path: str):
        self.labeled_pc_path = Path(labeled_pc_path)
        return 0

    def do_load_config(self, config_path: str, delim: str = "="):
        self.config = Config.from_file(config_path=Path(config_path), delim=delim)
        print(f"Loaded config with total {len(self.config.parameters)} parameters")
        return 0

    def do_load_executable(self, executable_path: str):
        if os.path.exists(executable_path):
            self.executable_path = executable_path
            print(f"Loaded executable {Path(executable_path).name}")
        else:
            raise RuntimeError(f"No executable at {executable_path}")

    def do_run_alg(self):
        if os.path.exists(self.output_folder):
            rmtree(self.output_folder)
        os.mkdir(self.output_folder)

        output_labels = self.output_folder / self.labels_name

        subprocess.run(
            [self.executable_path, self.pc_path, self.config.config_path, output_labels]
        )

        if os.path.exists(output_labels):
            labels_to_cloud(output_labels, self.pc_path, self.output_folder)
            print(f"Saved labeled cloud to {self.output_folder}")

    def do_init(self):
        init_list = {
            "pc": self.do_load_pc,
            "config": self.do_load_config,
            "executable": self.do_load_executable,
        }

        with open(self.__runner_cfg) as runner_config:
            for line in runner_config:
                p_name, value = line.strip().split("=")
                init_list[p_name](value)

    def get_command(self, command_name: str):
        if command_name in self.commands:
            return getattr(self, "do_" + command_name)
        else:
            raise RunnerException(f"Command '{command_name}' not found")

    def loop(self):
        while True:
            raw_input = input(self.prompt)
            command_name, args = self.parse(raw_input)
            try:
                self.get_command(command_name)(*args)
            except RunnerException as e:
                print(e.msg)

    @staticmethod
    def parse(raw_input: str):
        tokens = raw_input.split()
        return tokens[0], tokens[1:]
