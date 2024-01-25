# Print but only if currently above specified verbosity level
def verPrint(verbose_status, verbose_threshold, *args):
    if verbose_status >= verbose_threshold:
        print(args)
        print(' '.join([str(x) for x in args]))