# Print but only if currently above specified verbosity level
def verPrint(verbose_status, verbose_threshold, msg):
    if verbose_status >= verbose_threshold:
        print(verbose_status)
        print('threshold', verbose_threshold)
        print(msg)