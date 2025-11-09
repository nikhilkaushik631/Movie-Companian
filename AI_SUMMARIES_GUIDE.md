# AI Summaries for Trending Titles

This guide explains the AI summary system implemented for trending movie and TV show cards.

## Overview

The system automatically generates concise AI-powered summaries for trending titles using the following approach:

1. **Automatic Detection**: When new trending titles are loaded, the system checks if they have existing AI summaries
2. **Background Generation**: Missing summaries are generated automatically using the LLM chatbot
3. **Caching**: All summaries are cached in the database to avoid regenerating them
4. **UI Integration**: Summaries appear directly on trending cards with a clear AI indicator

## Key Features

### Backend Components

1. **New API Endpoints** (in `main.py`):
   - `POST /trending/summaries/generate` - Bulk generate summaries for trending titles
   - `GET /trending/summaries/check` - Check which titles need summaries
   - `GET /trending/summaries/{external_id}` - Get cached summary for specific title

2. **Database Storage**:
   - Uses existing `ContentSummary` table with `kind='card'` for trending card summaries
   - Stores by TMDB external ID for efficient lookup

3. **Smart Generation**:
   - Only generates for popular titles (popularity > 10)
   - Rate-limited to prevent API exhaustion
   - Uses specialized prompts for concise, engaging card summaries

### Frontend Integration

1. **Enhanced Trending Cards**:
   - Shows AI summaries directly on cards when available
   - Loading states while summaries are being generated
   - Manual "Get AI Summary" button for on-demand generation

2. **Automatic Loading**:
   - Loads summaries for first 3 items in each trending section
   - Checks for missing summaries on app load
   - Triggers bulk generation if many summaries are missing

3. **Visual Design**:
   - Clear AI indicator (🤖) to show content is AI-generated
   - Subtle styling that integrates with existing card design
   - Truncated display with ellipsis for longer summaries

## Usage

### For Users

1. **Automatic**: Summaries appear automatically on trending cards
2. **Manual**: Click "🤖 Get AI Summary" button on any card without a summary
3. **Detailed View**: Click the card to see full AI summary in the modal

### For Developers

#### Generate Summaries in Bulk

```bash
# Check which titles need summaries
python trending_summaries.py check

# Generate summaries for up to 10 titles
python trending_summaries.py generate 10

# Export summaries to JSON backup
python trending_summaries.py export summaries_backup.json

# Import summaries from JSON
python trending_summaries.py import summaries_backup.json

# Clean old summaries (older than 30 days)
python trending_summaries.py clean 30
```

#### API Usage

```javascript
// Check if summaries are needed
const checkResponse = await fetch('/trending/summaries/check');
const { missing_count, needs_generation } = await checkResponse.json();

// Generate summaries in bulk
if (needs_generation) {
    await fetch('/trending/summaries/generate', { method: 'POST' });
}

// Get specific summary
const summaryResponse = await fetch(`/trending/summaries/${tmdbId}`);
const { summary, cached } = await summaryResponse.json();
```

## Configuration

### Summary Generation Settings

- **Batch Size**: Generates up to 10 summaries per trending category
- **Rate Limiting**: Respects GROQ API limits with built-in delays
- **Cache Duration**: Summaries are cached indefinitely unless manually cleaned
- **Trigger Threshold**: Bulk generation triggers when >5 summaries are missing

### Customizing Summary Style

Edit the prompt in `trending_summaries.py` or the API endpoint:

```python
prompt = f"Write a compelling 2-3 sentence summary for {title}{year_part} [{type_label}] for a movie recommendation card. Focus on what makes it interesting - genre, key plot points, notable cast, or why it's trending. Keep it engaging and under 150 words."
```

## Technical Details

### Data Flow

1. **Trending Data Load**: TMDB API → Frontend State → Summary Check
2. **Summary Generation**: Backend LLM → Database Cache → Frontend Display
3. **Caching Strategy**: Database storage with external_id lookup

### Error Handling

- Graceful fallback when LLM is unavailable
- Retry logic for failed generations
- Rate limiting to prevent API exhaustion
- Silent failures that don't break the UI

### Performance Optimizations

- **Lazy Loading**: Only loads summaries for visible/priority items
- **Caching**: Database-backed caching prevents regeneration
- **Batch Operations**: Efficient bulk generation for multiple titles
- **Rate Limiting**: Prevents overwhelming external APIs

## Database Schema

The system uses the existing `ContentSummary` table:

```sql
CREATE TABLE content_summaries (
    id INTEGER PRIMARY KEY,
    title_external_id STRING,  -- TMDB ID
    title_source STRING,       -- 'tmdb'
    kind STRING,               -- 'card' for trending summaries
    text TEXT,                 -- The AI-generated summary
    created_at DATETIME
);
```

## Monitoring

### Health Checks

- Check summary generation success rate in logs
- Monitor rate limiting effectiveness
- Track cache hit rates

### Maintenance

- Regular cleanup of old summaries
- Backup summaries before major updates
- Monitor LLM API usage and costs

## Future Enhancements

1. **Summary Quality**: Implement summary rating system
2. **Personalization**: Generate summaries based on user preferences
3. **Multi-language**: Support summaries in different languages
4. **A/B Testing**: Test different summary styles and lengths
5. **Analytics**: Track which summaries lead to higher engagement

## Troubleshooting

### Common Issues

1. **No Summaries Generated**:
   - Check LLM API keys are configured
   - Verify chatbot initialization
   - Check rate limiting settings

2. **Slow Loading**:
   - Increase summary batch size
   - Pre-generate summaries during off-peak hours
   - Check database query performance

3. **Poor Summary Quality**:
   - Adjust the generation prompt
   - Try different LLM models
   - Add content filtering rules

### Debug Commands

```bash
# Check database for summaries
python -c "from main import *; db = SessionLocal(); print(db.query(ContentSummary).count())"

# Test summary generation manually
python trending_summaries.py generate 1

# Check API endpoint health
curl localhost:8000/trending/summaries/check
```

## Cost Considerations

- **LLM Usage**: Each summary costs ~1-2 API calls
- **Storage**: Minimal database storage cost
- **Rate Limiting**: Built-in protection against API overuse
- **Caching**: Reduces repeated generation costs

Monitor your LLM API usage dashboard and adjust generation frequency as needed.