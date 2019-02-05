from collections import OrderedDict
import multiprocessing as mp
import torch
import torchmed


class Sampler(object):
    def __init__(self, pattern_map, offset=1, nb_workers=1):
        # checking on pattern_map
        assert(isinstance(pattern_map, dict))
        assert(len(pattern_map) > 0)
        for k, v in pattern_map.items():
            assert(isinstance(k, str))
            assert(isinstance(v[0], str))
            assert(isinstance(v[1], torchmed.Pattern))

        # checking on nb_workers
        assert(isinstance(nb_workers, int))
        assert(nb_workers > 0)

        # checking on offset
        if isinstance(offset, list):
            assert(all(isinstance(n, int) for n in offset))
        else:
            assert(isinstance(offset, int))
            assert(offset > 0)

        self._coordinates = None
        self._nb_workers = nb_workers
        self._offset = offset
        self._pattern_map = OrderedDict(
            sorted(pattern_map.items(), key=lambda t: t[0]))

    def __len__(self):
        """Number of sampling coordinates."""
        return len(self._coordinates)

    def build(self, data):
        """Evaluates the valid sampling coordinates for all images, with respect
        to the pattern map and the data.

        Parameters
        ----------
        data : dict
            data dictionnary.


        .. note::
            the `data` dictionnary must contain at least one key named
            `image_ref` that we will be used a the main reference for image
            size.

        """
        from random import randint

        assert('image_ref' in data.keys())

        self._data = data
        for _, pattern_conf in self._pattern_map.items():
            input_name, pattern = pattern_conf
            pattern.prepare(self._data[input_name].to_torch())

        ref_image_size = self._data['image_ref'].to_torch().size()

        # set extraction offset
        if isinstance(self._offset, int):
            self._offset = [self._offset] * len(ref_image_size)
        if len(self._offset) != len(ref_image_size):
            raise ValueError('The offset dimensionality must equal '
                             'the image dimensionality.')

        patch_positions = [list(range(0, ref_image_size[dim], offset))
                           for dim, offset in enumerate(self._offset)]

        # eval the optimal split size of dim 0, useful for large images
        first_axis_size = len(patch_positions[0])
        ideal_split_size = int(first_axis_size // self._nb_workers)
        split_size = randint(ideal_split_size // 3, ideal_split_size // 2)

        # if split_size == 0 the default split size is 1
        split_size = split_size if split_size > 0 else 1

        splits = []
        split_index = 0
        while split_index < first_axis_size:
            # if the next split is too big for the dataset, resize it
            if split_index + split_size > first_axis_size:
                split_size = first_axis_size - split_index

            splits.append((split_index, split_index + split_size))
            split_index += split_size

        # if we have more than one worker we can use multiprocessing/queues
        if self._nb_workers > 1:
            # create queues
            task_queue = mp.JoinableQueue()
            done_queue = mp.Queue()

            for split in splits:
                # give the correct extraction positions to the worker
                patch_pos = (patch_positions[0][split[0]: split[1]],
                             *patch_positions[1:])
                task_queue.put(patch_pos)
            # add None objects so workers can stop when there is no more work
            [task_queue.put(None) for i in range(0, self._nb_workers - 1)]

            # start worker processes
            producers = []
            end_events = [mp.Event() for i in range(0, self._nb_workers - 1)]
            for i in range(0, self._nb_workers - 1):
                process = mp.Process(target=self._sampler_worker,
                                     args=(task_queue,
                                           done_queue,
                                           end_events[i])
                                     )
                process.start()
                producers.append(process)

            """
            Read results from workers and wait until the end of all.
            Each worker returns None when it has ended, so we need to count
            how many None we received before merging all the
            result arrays together.
            """
            result_arrays = []
            nb_ended_workers = 0
            while nb_ended_workers != self._nb_workers - 1:
                worker_result = done_queue.get()
                if worker_result is None:
                    nb_ended_workers += 1
                else:
                    result_arrays.append(worker_result)

            # concatenates all the results
            if len(result_arrays) == 0:
                self._coordinates = torch.ShortTensor(0, 0)
            else:
                self._coordinates = torch.cat(result_arrays, 0)

            # we can set free all the background processes
            [end_events[i].set() for i in range(0, self._nb_workers - 1)]

            # at this point all the processes are already ended, we can close
            # them by calling join on each one, and terminate them properly
            for process in producers:
                process.join()

            # Join and close queues
            done_queue.close()
            done_queue.join_thread()
            task_queue.close()
            task_queue.join_thread()

        # if one process: evaluate all the splits one after the other and concat
        else:
            for split in splits:
                patch_pos = (patch_positions[0][split[0]: split[1]],
                             *patch_positions[1:])
                coords, nb_elems = self.get_positions(patch_pos)
                if self._coordinates is None and nb_elems > 0:
                    self._coordinates = coords[0:nb_elems, ]
                elif nb_elems > 0:
                    self._coordinates = torch.cat(
                        (self._coordinates, coords[0:nb_elems, ]), 0)

    def _sampler_worker(self, task_queue, done_queue, end_event):
        """
        Each worker is given a task queue to read from, and a result queue to
        write results in. The worker is given the positions of extraction,
        then it returns a result array, picks up a new task and so on. Once the
        worker picks a None object it means there is no more work to do,
        thus the worker loop is interrupted.
        """
        worker_result = None
        while True:
            task_args = task_queue.get()
            if task_args is None:
                task_queue.task_done()
                if worker_result is not None:
                    done_queue.put(worker_result)
                done_queue.put(None)
                end_event.wait()
                break
            else:
                coords, nb_elems = self.get_positions(task_args)

                if worker_result is None and nb_elems > 0:
                    worker_result = coords[0:nb_elems, ]
                elif nb_elems > 0:
                    worker_result = torch.cat(
                        (worker_result, coords[0:nb_elems, ]), 0)

                task_queue.task_done()
