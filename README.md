the cukiverb
===========
Cukiverb is a real-time convolution reverb for the Digital Audio Workstation (DAW) Reaper.
It is implemented in Jesusonic programming lenguage (see www.cockos.com/jesusonic). The 
implementation uses particioned convolution following overlap-add method. We also provide 
a python script for automatically measure room impulse responses for using with Cukiverb. This script can be called from Reaper through the ReaScript (see www.reaper.fm/sdkreascript/reascript.php) tool. 

Install
=========

For using Cukiverb you must have installed Reaper. 

* copy 'cukiverb' (make sure it does not have an extension) to your Reaper JS effects folder.
* copy 'ir_measrement.py' to your REAPER/Scripts folder.

Usage::
-------------
* Open Reaper
* Load 'cukiverb' from the JS fx list

