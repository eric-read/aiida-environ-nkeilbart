# -*- coding: utf-8 -*-
from typing import Tuple

from aiida.orm import Data


class EnvironDielectric:
    def __init__(self, **kwargs):
        self._static_permittivity = None
        self._optical_permittivity = None
        self._position = None
        self._width = None
        self._spread = None
        self._dim = None
        self._axis = None

        if "environ_dielectric" in kwargs:
            environ_dielectric = kwargs.pop("environ_dielectric")
            if kwargs:
                raise ValueError(
                    "If you pass 'environ_dielectric', you cannot pass any further parameter to the "
                    "EnvironDielectric constructor"
                )
            if not isinstance(environ_dielectric, EnvironDielectric):
                raise ValueError("'environ_dielectric' must be of type EnvironDielectric")
            self.static_permittivity = environ_dielectric.static_permittivity
            self.optical_permittivity = environ_dielectric.optical_permittivity
            self.position = environ_dielectric.position
            self.width = environ_dielectric.width
            self.spread = environ_dielectric.spread
            self.dim = environ_dielectric.dim
            self.axis = environ_dielectric.axis
        elif "raw" in kwargs:
            raw = kwargs.pop("raw")
            if kwargs:
                raise ValueError(
                    "If you pass 'raw', you cannot pass any further parameter to the Site constructor"
                )
            try:
                self.static_permittivity = raw["static_permittivity"]
                self.optical_permittivity = raw["optical_permittivity"]
                self.position = raw["position"]
                self.width = raw['width']
                self.spread = raw["spread"]
                self.dim = raw["dim"]
                self.axis = raw["axis"]
            except KeyError as exc:
                raise ValueError(
                    f"Invalid raw object, it does not contain any key {exc.args[0]}"
                )
            except TypeError:
                raise ValueError("Invalid raw object, it is not a dictionary")
        else:
            try:
                self.static_permittivity = kwargs.pop("static_permittivity")
                self.optical_permittivity = kwargs.pop("optical_permittivity")
                self.position = kwargs.pop("position")
                self.width = kwargs.pop("width")
                self.spread = kwargs.pop("spread")
                self.dim = kwargs.pop("dim")
                self.axis = kwargs.pop("axis")
            except KeyError as exc:
                raise ValueError(f"You need to specify {exc.args[0]}")
            if kwargs:
                raise ValueError(f"Unrecognized parameters: {kwargs.keys}")

    def get_raw(self):
        """
        Return the raw version of the site, mapped to a suitable dictionary.
        This is the format that is actually used to store each site of the
        structure in the DB.

        :return: a python dictionary with the site.
        """
        return {
            "static_permittivity": self.static_permittivity,
            "optical_permittivity": self.optical_permittivity,
            "position": self.position,
            "width": self.width,
            "spread": self.spread,
            "dim": self.dim,
            "axis": self.axis,
        }

    @property
    def static_permittivity(self):
        """Return the static permittivity

        Returns:
            float: static permittivity
        """
        return self._static_permittivity

    @static_permittivity.setter
    def static_permittivity(self, value: float):
        """Set the static permittivity

        Args:
            value (float): static permittivity
        """
        self._static_permittivity = float(value)

    @property
    def optical_permittivity(self):
        """Return the optical permittivity

        Returns:
            float: optical permittivity
        """
        return self._static_permittivity

    @optical_permittivity.setter
    def optical_permittivity(self, value: float):
        """Set the optical permittivity

        Args:
            value (float): optical permittivity
        """
        self._optical_permittivity = float(value)

    @property
    def position(self):
        """Return the position

        Returns:
            Tuple[float]: coordinates in angstrom
        """
        return self._position

    @position.setter
    def position(self, value: Tuple[float]):
        """Set the position

        Args:
            value (Tuple[float]): coordinates in angstrom

        Raises:
            ValueError: length of tuple incorrect
            ValueError: list/tuple not provided
        """
        try:
            internal_pos = tuple(float(i) for i in value)
            if len(internal_pos) != 3:
                raise ValueError
        # value is not iterable or elements are not floats or len != 3
        except (ValueError, TypeError):
            raise ValueError(
                "Wrong format for position, must be a list of three float numbers."
            )
        self._position = internal_pos

    @property
    def width(self):
        """Return the width of the dielectric region

        Returns:
            float: width
        """
        return self._width

    @width.setter
    def width(self, value: float):
        """Set the width of the dielectric region

        Args:
            value (float): width
        """
        self._width = float(value)

    @property
    def spread(self):
        """Return the spread of the dielectric region

        Returns:
            float: spread
        """
        return self._spread

    @spread.setter
    def spread(self, value: float):
        """Set the spread of the dielectric region

        Args:
            value (float): spread
        """
        self._spread = float(value)

    @property
    def dim(self):
        """Return the dimensionality of the dielectric object

        Returns:
            int: dimensionality (0-2)
        """
        return self._dim

    @dim.setter
    def dim(self, value: int):
        """Set the dimensionality of the dielectric region

        Args:
            value (int): dimensionality (0-2)
        """
        if value < 0 or value > 2:
            raise ValueError("Dimensionality must be between 0 and 2")
        self._dim = int(value)

    @property
    def axis(self):
        """Return the axis (1-3), where x=1, y=2, z=3

        If dim=2, the axis is orthogonal to the 2D plane
        If dim=1, the axis is along the 1D direction

        Returns:
            int: axis
        """
        return self._axis

    @axis.setter
    def axis(self, value: int):
        """Return the axis (1-3), where x=1, y=2, z=3

        If dim=2, the axis is orthogonal to the 2D plane
        If dim=1, the axis is along the 1D direction

        Args:
            value (int): axis
        """
        if value < 1 or value > 3:
            raise ValueError("Axis must be between 1 and 3")
        self._axis = int(value)

    def __repr__(self):
        return f"<{self.__class__.__name__}: {str(self)}>"

    def __str__(self):
        ax = {1: "x", 2: "y", 3: "z"}
        return (
            f"static permittivity '{self.static_permittivity}', optical permittivity '{self.optical_permittivity}' "
            f"@ {self.position[0]},{self.position[1]},{self.position[2]}"
            f" (dim {self.dim}, '{ax[self.axis]}' axis, spread={self.spread})"
        )

    @static_permittivity.setter
    def static_permittivity(self, value):
        self._static_permittivity = value

    @optical_permittivity.setter
    def optical_permittivity(self, value):
        self._optical_permittivity = value


class EnvironDielectricData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def append_dielectric(
        self, static_permittivity: float, optical_permittivity: float, position: Tuple[float], width: float, spread: float, dim: int, axis: int
    ):
        dielectric = EnvironDielectric(
            static_permittivity=static_permittivity,
            optical_permittivity=optical_permittivity,
            position=position,
            width=width,
            spread=spread,
            dim=dim,
            axis=axis
        )
        self.attributes.setdefault("environ_dielectrics", []).append(dielectric.get_raw())

    def clear_dielectrics(self):
        """
        Removes all dielectric regions for the EnvironDielectricData object.
        """
        from aiida.common.exceptions import ModificationNotAllowed

        if self.is_stored:
            raise ModificationNotAllowed(
                "The EnvironDielectricData object cannot be modified, it has already been stored"
            )

        self.set_attribute("environ_dielectrics", [])

    @property
    def environ_dielectrics(self):
        """
        Returns a list of sites.
        """
        try:
            raw_dielectrics = self.get_attribute("environ_dielectrics")
        except AttributeError:
            raw_dielectrics = []
        return [EnvironDielectric(raw=i) for i in raw_dielectrics]

    def environ_output(self):
        """Prints out to string for `environ.in`"""

        environ_dielectrics = self.environ_dielectrics
        if len(environ_dielectrics) == 0:
            # nothing here, just return empty string
            return ""

        # TODO add support for other units
        inputappend = "DIELECTRIC_REGIONS (angstrom)\n"
        for dielectric in environ_dielectrics:
            inputappend += (
                f"{dielectric.static_permittivity} {dielectric.optical_permittivity} "
                f"{dielectric.position[0]:10.6f} {dielectric.position[1]:10.6f} {dielectric.position[2]:10.6f} "
                f"{dielectric.width:10.6f} {dielectric.spread:10.6f} {dielectric.dim:d} {dielectric.axis:d}\n"
            )

        return inputappend

    def __len__(self):
        return len(self.environ_dielectrics)
