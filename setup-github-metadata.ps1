# GitHub Repository Setup Script - Auto-loads token from .env
# Sets description, topics, and website for your repo

# Load .env file
$envFile = "Assignment_r\.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^GIT_TOKEN=(.+)$') {
            $token = $matches[1].Trim()
        }
    }
}

if (-not $token) {
    Write-Host "❌ GIT_TOKEN not found in Assignment_r\.env" -ForegroundColor Red
    Write-Host "Please add: GIT_TOKEN=your_github_token_here" -ForegroundColor Yellow
    exit
}

$repo = "ysocrius/DataWeave_TurerZ"

$headers = @{
    "Authorization" = "token $token"
    "Accept"        = "application/vnd.github.v3+json"
}

try {
    # Set repository description and website
    $repoData = @{
        description = "🚀 DataWeave - AI-powered document processor that transforms unstructured PDFs into structured Excel files with 100% data fidelity. Features intelligent deduplication, self-improving learning system, and modern React UI."
        homepage    = ""
    } | ConvertTo-Json

    Write-Host "Setting repository description..." -ForegroundColor Cyan
    $response = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo" -Method PATCH -Headers $headers -Body $repoData -ContentType "application/json"
    Write-Host "✅ Description updated!" -ForegroundColor Green

    # Set repository topics
    $topics = @{
        names = @(
            "ai",
            "document-processing",
            "pdf-to-excel",
            "machine-learning",
            "python",
            "react",
            "typescript",
            "fastapi",
            "openai",
            "gpt4",
            "data-extraction",
            "automation",
            "nlp",
            "mongodb",
            "full-stack"
        )
    } | ConvertTo-Json

    Write-Host "Setting repository topics..." -ForegroundColor Cyan
    $topicResponse = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/topics" -Method PUT -Headers $headers -Body $topics -ContentType "application/json"
    Write-Host "✅ Topics updated!" -ForegroundColor Green

    Write-Host "`n🎉 Repository metadata updated successfully!" -ForegroundColor Green
    Write-Host "View at: https://github.com/$repo" -ForegroundColor Cyan

}
catch {
    Write-Host "`n❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Make sure your GIT_TOKEN has 'repo' scope" -ForegroundColor Yellow
}
