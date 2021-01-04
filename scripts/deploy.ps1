python setup.py sdist bdist_wheel
$version="0.3.3"
$files_to_handle_str="dist/time_series_models-$version*" 
twine check $files_to_handle_str
twine upload $files_to_handle_str