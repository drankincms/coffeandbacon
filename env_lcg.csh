# http://lcginfo.cern.ch/release/95apython3/
# Try to guess SL6 vs. CC7
uname -r | grep -q el6
set isel6 = $?
if ( $isel6 == 0 ) then
  source /cvmfs/sft.cern.ch/lcg/views/LCG_95apython3/x86_64-slc6-gcc8-opt/setup.csh
else
  source /cvmfs/sft.cern.ch/lcg/views/LCG_95apython3/x86_64-centos7-gcc8-opt/setup.csh
endif

setenv PYTHONPATH "~/.local/lib/python3.6/site-packages:"$PYTHONPATH
