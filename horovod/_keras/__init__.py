# Copyright 2017 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import horovod.tensorflow as hvd
import tensorflow as tf


def create_distributed_optimizer(keras, optimizer, name, device_dense, device_sparse,
                                 compression, sparse_as_dense):
    class _DistributedOptimizer(keras.optimizers.Optimizer):
        def __init__(self, name, device_dense, device_sparse, compression, sparse_as_dense,
                     config):
            if name is None:
                name = "Distributed%s" % self.__class__.__base__.__name__
            self._name = name
            self._device_dense = device_dense
            self._device_sparse = device_sparse
            self._compression = compression
            self._sparse_as_dense = sparse_as_dense
            super(self.__class__, self).__init__(**config)

        def get_gradients(self, loss, params):
            """
            Compute gradients of all trainable variables.

            See Optimizer.get_gradients() for more info.

            In DistributedOptimizer, get_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            gradients = super(self.__class__, self).get_gradients(loss, params)
            if hvd.size() > 1:
                tf.compat.v1.logging.info("I'M GETTING THE GRADIENTS!")
                averaged_gradients = []
                with tf.name_scope(self._name + "_Allreduce"):
                    gradients = maybe_rename_grads(gradients)
                    for grad in gradients:
                        if grad is not None:
                            if self._sparse_as_dense and \
                                    isinstance(grad, tf.IndexedSlices):
                                grad = tf.convert_to_tensor(grad)
                            tf.compat.v1.logging.info("tensor = %s" % grad.name)
                            avg_grad = hvd.allreduce(grad,
                                                     device_dense=self._device_dense,
                                                     device_sparse=self._device_sparse,
                                                     compression=self._compression)
                            averaged_gradients.append(avg_grad)
                        else:
                            averaged_gradients.append(None)
                    return averaged_gradients
            else:
                return gradients

    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override get_gradients() method with an allreduce implementation.
    # This class will have the same name as the optimizer it's wrapping, so that the saved
    # model could be easily restored without Horovod.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(name, device_dense, device_sparse, compression, sparse_as_dense,
               optimizer.get_config())

# HACK: for some reason the batch norm gradients have different names across
# different machines and horovod doesn't know how to average them together.
# Here we rename them so they have the same name across ranks.
def maybe_rename_grads(grads):
    tf.compat.v1.logging.info("Renaming gradients!")
    # First, collect all the ones with Identity
    # Map grad prefix to index in original grads array
    identity_grad_indices = {}
    for i, grad in enumerate(grads):
        if "Identity" in grad.name:
            split = grad.name.split("Identity")
            prefix = split[0]
            if prefix not in identity_grad_indices:
                identity_grad_indices[prefix] = []
            identity_grad_indices[prefix].append(i)
    # Now, for each prefix, we rename the grads according to their "rank"
    # For example, "Identity:0" has a rank of 0, "Identity_1:0" has a rank of 1
    for prefix in identity_grad_indices:
        ranks = [grads[i].name.split("Identity")[1] for i in identity_grad_indices[prefix]]
        ranks = [0 if "_" not in rank else int(rank.replace("_", "").split(":")[0]) for rank in ranks]
        import numpy as np
        rank_indices = np.argsort(ranks)
        for k, j in enumerate(rank_indices):
            i = identity_grad_indices[prefix][j]
            new_name = "%sIdentity_%s:0" % (grads[i].name.split("Identity")[0], k)
            grads[i] = tf.identity(grads[i], name=new_name)
    return grads

def broadcast_global_variables(backend, root_rank):
    bcast_op = hvd.broadcast_global_variables(root_rank)
    return backend.get_session().run(bcast_op)


def allreduce(backend, value, name, average):
    allreduce_op = hvd.allreduce(tf.constant(value, name=name), average=average)
    return backend.get_session().run(allreduce_op)


def allgather(backend, value, name):
    allgather_op = hvd.allgather(tf.constant(value, name=name))
    return backend.get_session().run(allgather_op)


def broadcast(backend, value, root_rank, name):
    bcast_op = hvd.broadcast(tf.constant(value, name=name), root_rank)
    return backend.get_session().run(bcast_op)


def load_model(keras, wrap_optimizer, filepath, custom_optimizers, custom_objects):
    horovod_objects = {
        subclass.__name__.lower(): wrap_optimizer(subclass)
        for subclass in keras.optimizers.Optimizer.__subclasses__()
        if subclass.__module__ == keras.optimizers.Optimizer.__module__
    }

    if custom_optimizers is not None:
        horovod_objects.update({
            cls.__name__: wrap_optimizer(cls)
            for cls in custom_optimizers
        })

    if custom_objects is not None:
        horovod_objects.update(custom_objects)

    return keras.models.load_model(filepath, custom_objects=horovod_objects)
