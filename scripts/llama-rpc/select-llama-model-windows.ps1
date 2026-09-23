$script:RecommendedHuggingFaceRepository = 'bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:Q4_K_M'

function Get-LlamaModelSearchDirectory {
    param(
        [string]$LlamaCppDirectory
    )

    $huggingFaceCacheDirectory = $env:HF_HUB_CACHE
    if (-not $huggingFaceCacheDirectory) {
        if ($env:HF_HOME) {
            $huggingFaceCacheDirectory = Join-Path $env:HF_HOME 'hub'
        }
        else {
            $huggingFaceCacheDirectory = Join-Path $HOME '.cache\huggingface\hub'
        }
    }

    return @(
        (Join-Path $HOME '.lmstudio\models')
        (Join-Path $HOME '.ollama\models')
        $huggingFaceCacheDirectory
        (Join-Path $LlamaCppDirectory 'models')
    )
}

function Resolve-LlamaModelSelection {
    param(
        [string]$ModelPath,
        [string]$HuggingFaceRepository,
        [string]$LlamaCppDirectory
    )

    if ($ModelPath -or $HuggingFaceRepository) {
        return [pscustomobject]@{
            ModelPath             = $ModelPath
            HuggingFaceRepository = $HuggingFaceRepository
        }
    }

    if ([Console]::IsInputRedirected) {
        return [pscustomobject]@{
            ModelPath             = ''
            HuggingFaceRepository = $script:RecommendedHuggingFaceRepository
        }
    }

    $discoveredModelPaths = @()
    foreach ($searchDirectory in (Get-LlamaModelSearchDirectory -LlamaCppDirectory $LlamaCppDirectory)) {
        if (-not (Test-Path -LiteralPath $searchDirectory)) {
            continue
        }
        $discoveredModelPaths += Get-ChildItem -LiteralPath $searchDirectory -Recurse -File -Filter '*.gguf' -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty FullName
    }
    $discoveredModelPaths = $discoveredModelPaths | Sort-Object -Unique

    if ($discoveredModelPaths.Count -eq 0) {
        Write-Warning "No downloaded GGUF models found; using $script:RecommendedHuggingFaceRepository."
        return [pscustomobject]@{
            ModelPath             = ''
            HuggingFaceRepository = $script:RecommendedHuggingFaceRepository
        }
    }

    Write-Host 'Select a llama.cpp model:'
    for ($index = 0; $index -lt $discoveredModelPaths.Count; $index++) {
        Write-Host ("  {0}) {1}" -f ($index + 1), $discoveredModelPaths[$index])
    }
    $downloadOptionNumber = $discoveredModelPaths.Count + 1
    Write-Host ("  {0}) Download recommended: {1}" -f $downloadOptionNumber, $script:RecommendedHuggingFaceRepository)

    while ($true) {
        $selection = Read-Host 'Model number'
        $selectedNumber = 0
        if (-not [int]::TryParse($selection, [ref]$selectedNumber)) {
            Write-Warning 'Invalid selection.'
            continue
        }
        if ($selectedNumber -eq $downloadOptionNumber) {
            return [pscustomobject]@{
                ModelPath             = ''
                HuggingFaceRepository = $script:RecommendedHuggingFaceRepository
            }
        }
        if ($selectedNumber -ge 1 -and $selectedNumber -le $discoveredModelPaths.Count) {
            return [pscustomobject]@{
                ModelPath             = $discoveredModelPaths[$selectedNumber - 1]
                HuggingFaceRepository = ''
            }
        }
        Write-Warning 'Invalid selection.'
    }
}
