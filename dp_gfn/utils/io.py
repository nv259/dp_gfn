"""
MIT License

Copyright (c) 2022 Tristan Deleu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import haiku as hk

from typing import Mapping


def save(filename, **trees):
    data = {}
    for prefix, tree in trees.items():
        if isinstance(tree, Mapping):
            tree = hk.data_structures.to_haiku_dict(tree)
            for module_name, name, value in hk.data_structures.traverse(tree):
                data[f'{prefix}/{module_name}/{name}'] = value
        else:
            data[prefix] = tree

    np.savez(filename, **data)


def load(filename, **kwargs):
    data = {}
    f = open(filename, 'rb') if isinstance(filename, str) else filename
    results = np.load(f, **kwargs)

    for key in results.files:
        prefix, delimiter, name = key.rpartition('/')
        if delimiter:
            prefix, _, module_name = prefix.partition('/')
            if prefix not in data:
                data[prefix] = {}
            if module_name not in data[prefix]:
                data[prefix][module_name] = {}
            data[prefix][module_name][name] = results[key]
        else:
            data[name] = results[key]

    for prefix, tree in data.items():
        if isinstance(tree, dict):
            data[prefix] = hk.data_structures.to_haiku_dict(tree)
    
    if isinstance(filename, str):
        f.close()
    del results

    return data