while len(pending_task) > 0 and pending_task[0].ready():
            print(len(pending_task))
            res, debug = pending_task.popleft().get()
            cv2.imshow('result', res)
            if debug is not None:
                cv2.imshow('debug', debug)    