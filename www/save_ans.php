<?php
$filename = sprintf("%s_%s.save", $_POST["task"],$_POST["fileId"]);
echo $filename . "\n";

$F = fopen($filename, 'w');
fprintf($F, "%s\n", $_POST["ans"]);
fclose($F);
?>
