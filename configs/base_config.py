import ast
import six
import copy
import json
import yaml
import collections


def eval_str_fn(val):
    if val.lower() in {"true", "false"}:
        return val.lower() == "true"

    try:
        return ast.literal_eval(val)
    except ValueError:
        return val


class Config(object):
    def __init__(self, config_dict=None):
        self.update(config_dict)

    def __setattr__(self, k, v):
        self.__dict__[k] = Config(v) if isinstance(v, dict) else copy.deepcopy(v)
    
    def __getattr__(self, k):
        return self.__dict__[k]
    
    def __repr__(self):
        return repr(self.as_dict())
    
    def __str__(self):
        try:
            return json.dumps(self.as_dict(), indent=4)
        except TypeError:
            return str(self.as_dict())
    
    def _update(self, config_dict, allow_new_keys=True):
        """Recursively update internal members."""
        if not config_dict:
            return 
        
        for k, v in six.iteritems(config_dict):
            if k not in self.__dict__.keys():
                if allow_new_keys:
                    self.__setattr__(k, v)
                else:
                    raise KeyError("Key `{}` does not exist for overriding.".format(k))
            else:
                if isinstance(v, dict):
                    self.__dict__[k]._update(v, allow_new_keys)
                else:
                    self.__dict__[k] = copy.deepcopy(v)
    
    def get(self, k, default_value=None):
        return self.__dict__.get(k, default_value)
    
    def update(self, config_dict):
        """Update members while allowing new keys."""
        self._update(config_dict, allow_new_keys=True)
    
    def override(self, config_dict_or_str, allow_new_keys=False):
        if isinstance(config_dict_or_str, str):
            if not config_dict_or_str:
                return
            elif '=' in config_dict_or_str:
                config_dict = self.parse_from_str(config_dict_or_str)
            elif config_dict_or_str.endswith('.yaml'):
                config_dict = self.parse_from_yaml(config_dict_or_str)
            else:
                raise ValueError(
                    'Invalid string {}, must end with .yaml or contains "=".'.format(
                        config_dict_or_str))
        elif isinstance(config_dict_or_str, dict):
          config_dict = config_dict_or_str
        else:
          raise ValueError('Unknown value type: {}'.format(config_dict_or_str))

        self._update(config_dict, allow_new_keys)
    
    def parse_from_yaml(self, yaml_path):
        """Parses a yaml file and returns a dictionary."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
            return config_dict
    
    def save_to_yaml(self, yaml_path):
        """Write a dict into a yaml file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.as_dict(), f, default_flow_style=False)
    
    def parse_from_str(self, config_str):
        """Parse a string like 'x.y=1,x.z=2' to nested dict {x: {y: 1, z: 2}}."""
        if not config_str:
            return {}
        config_dict = {}
        try:
            for kv_pair in config_str.split(','):
                if not kv_pair:  # skip empty string
                    continue
                key_str, value_str = kv_pair.split('=')
                key_str = key_str.strip()

                def add_kv_recursive(k, v):
                    """Recursively parse x.y.z=tt to {x: {y: {z: tt}}}."""
                    if '.' not in k:
                        if '*' in v:
                            # we reserve * to split arrays.
                            return {k: [eval_str_fn(vv) for vv in v.split('*')]}
                        return {k: eval_str_fn(v)}
                    pos = k.index('.')
                    return {k[:pos]: add_kv_recursive(k[pos + 1:], v)}

                def merge_dict_recursive(target, src):
                    """Recursively merge two nested dictionary."""
                    for k in src.keys():
                        if ((k in target and isinstance(target[k], dict) and
                            isinstance(src[k], collections.abc.Mapping))):
                            merge_dict_recursive(target[k], src[k])
                        else:
                            target[k] = src[k]

                merge_dict_recursive(config_dict, add_kv_recursive(key_str, value_str))
            return config_dict
        except ValueError:
            raise ValueError('Invalid config_str: {}'.format(config_str))
    
    def as_dict(self):
        """Returns a dict representation."""
        config_dict = {}
        for k, v in six.iteritems(self.__dict__):
            if isinstance(v, Config):
                config_dict[k] = v.as_dict()
            else:
                config_dict[k] = copy.deepcopy(v)
        
        return config_dict

