import wfdb
record = wfdb.rdrecord('100')  # просто '100', без 'mitdb/'
print(record.sig_name)
print(record.fs)
print("OK!")