import multiprocessing as mp
import os


def parallelize_system_calls(nb_workers, commands):
    # create queues
    task_queue = mp.JoinableQueue()
    done_queue = mp.Queue()

    if isinstance(commands, (list, tuple)):
        for command in commands:
            task_queue.put(command)
    else:
        task_queue.put(commands)

    # add None objects so workers can stop when there is no more work
    [task_queue.put(None) for i in range(0, nb_workers - 1)]

    # start worker processes
    producers = []
    end_events = [mp.Event() for i in range(0, nb_workers - 1)]
    for i in range(0, nb_workers - 1):
        process = mp.Process(target=sampler_worker,
                             args=(task_queue,
                                   done_queue,
                                   end_events[i])
                             )
        process.start()
        producers.append(process)

    nb_ended_workers = 0
    while nb_ended_workers != nb_workers - 1:
        worker_result = done_queue.get()
        if worker_result is None:
            nb_ended_workers += 1

    # we can set free all the background processes
    [end_events[i].set() for i in range(0, nb_workers - 1)]

    # at this point all the processes are already ended, we can close
    # them by calling join on each one, and terminate them properly
    for process in producers:
        process.join()

    # Join and close queues
    done_queue.close()
    done_queue.join_thread()
    task_queue.close()
    task_queue.join_thread()


def sampler_worker(task_queue, done_queue, end_event):
    """
    Each worker is given a task queue to read from, and a result queue to
    write results in. The worker is given the positions of extraction,
    then it returns a result array, picks up a new task and so on. Once the
    worker picks a None object it means there is no more work to do,
    thus the worker loop is broken.
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
            os.system(task_args)
            task_queue.task_done()
