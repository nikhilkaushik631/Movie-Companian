#!/usr/bin/env python3
"""
Trending Summaries Management Script

This script helps manage AI summaries for trending titles:
1. Check which trending titles need summaries
2. Generate summaries in bulk
3. Export/import summaries to/from JSON
4. Clean up old summaries

Usage:
    python trending_summaries.py check        # Check missing summaries
    python trending_summaries.py generate     # Generate missing summaries
    python trending_summaries.py export       # Export summaries to JSON
    python trending_summaries.py import file  # Import summaries from JSON
    python trending_summaries.py clean        # Clean old summaries
"""

import sys
import json
import asyncio
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Import from main app
try:
    from main import (
        SessionLocal, Title, ContentSummary,
        get_cached_summary, save_cached_summary,
        TMDB_BASE_URL, TMDB_API_KEY, TMDB_BEARER,
        rate_limiter
    )
    from summary_generator import generate_ai_summary, get_summary_generator
except ImportError:
    print("Error: Could not import from main.py. Make sure you're running this from the project root.")
    sys.exit(1)


class TrendingSummaryManager:
    def __init__(self):
        self.db = SessionLocal()
        self.trending_endpoints = [
            '/trending/movie/day', '/trending/movie/week', '/movie/popular', '/movie/top_rated',
            '/trending/tv/day', '/trending/tv/week', '/tv/popular', '/tv/top_rated'
        ]

    def close(self):
        self.db.close()

    def check_missing_summaries(self):
        """Check which trending titles are missing AI summaries"""
        print("🔍 Checking for missing AI summaries...")

        missing_count = 0
        total_titles = 0

        # Get recent popular titles that might need summaries
        titles = (
            self.db.query(Title)
            .filter(Title.source == 'tmdb')
            .filter(Title.popularity > 5)
            .order_by(Title.updated_at.desc())
            .limit(100)
            .all()
        )

        print(f"Checking {len(titles)} popular titles...")

        for title in titles:
            total_titles += 1

            # Check for card summary
            card_summary = get_cached_summary(self.db, title.external_id, 'tmdb', 'card')
            # Check for general LLM summary
            llm_summary = get_cached_summary(self.db, title.external_id, 'tmdb', 'llm')

            if not card_summary and not llm_summary:
                missing_count += 1
                print(f"  ❌ Missing: {title.title} ({title.year}) - ID: {title.external_id}")

        print(f"\n📊 Summary: {missing_count}/{total_titles} titles missing AI summaries")
        return missing_count, total_titles

    async def generate_missing_summaries(self, limit=10):
        """Generate AI summaries for trending titles using dedicated LLM with TMDB plot data"""
        summary_gen = get_summary_generator()
        if not summary_gen:
            print("❌ Error: Summary generator not initialized. Check your GROQ API key.")
            return 0

        print(f"🤖 Generating up to {limit} missing AI summaries using TMDB plot data...")

        # Get titles that need summaries
        titles = (
            self.db.query(Title)
            .filter(Title.source == 'tmdb')
            .filter(Title.popularity > 10)
            .order_by(Title.popularity.desc())
            .limit(50)
            .all()
        )

        generated_count = 0

        for title in titles:
            if generated_count >= limit:
                break

            # Check if summary already exists
            existing = get_cached_summary(self.db, title.external_id, 'tmdb', 'card')
            if existing:
                continue

            try:
                print(f"Generating summary for: {title.title} ({title.year})")

                # Apply rate limiting for LLM
                await rate_limiter.wait_if_needed("groq")

                # Generate summary using dedicated LLM with TMDB plot data
                media_type = title.media_type or 'movie'
                summary = await generate_ai_summary(
                    external_id=title.external_id,
                    title=title.title,
                    media_type=media_type,
                    summary_type="card"
                )

                # Save summary
                save_cached_summary(self.db, title.external_id, 'tmdb', summary, 'card')
                generated_count += 1
                print(f"  ✅ Generated and saved summary with TMDB plot data")

                # Small delay to prevent overwhelming the LLM
                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"  ❌ Error generating summary: {e}")
                continue

        print(f"\n🎉 Generated {generated_count} new AI summaries using TMDB plot data")
        return generated_count

    def export_summaries(self, filename="trending_summaries.json"):
        """Export all summaries to JSON file"""
        print(f"📤 Exporting summaries to {filename}...")

        summaries = (
            self.db.query(ContentSummary)
            .filter(ContentSummary.kind.in_(['card', 'llm']))
            .all()
        )

        export_data = []
        for summary in summaries:
            export_data.append({
                'title_source': summary.title_source,
                'title_external_id': summary.title_external_id,
                'kind': summary.kind,
                'text': summary.text,
                'created_at': summary.created_at.isoformat()
            })

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Exported {len(export_data)} summaries to {filename}")
        return len(export_data)

    def import_summaries(self, filename):
        """Import summaries from JSON file"""
        if not Path(filename).exists():
            print(f"❌ Error: File {filename} not found")
            return 0

        print(f"📥 Importing summaries from {filename}...")

        with open(filename, 'r', encoding='utf-8') as f:
            import_data = json.load(f)

        imported_count = 0

        for item in import_data:
            try:
                # Check if summary already exists
                existing = (
                    self.db.query(ContentSummary)
                    .filter(
                        ContentSummary.title_external_id == item['title_external_id'],
                        ContentSummary.title_source == item['title_source'],
                        ContentSummary.kind == item['kind']
                    )
                    .first()
                )

                if existing:
                    # Update existing
                    existing.text = item['text']
                    existing.created_at = datetime.now()
                else:
                    # Create new
                    summary = ContentSummary(
                        title_source=item['title_source'],
                        title_external_id=item['title_external_id'],
                        kind=item['kind'],
                        text=item['text']
                    )
                    self.db.add(summary)

                imported_count += 1

            except Exception as e:
                print(f"Error importing summary: {e}")
                continue

        self.db.commit()
        print(f"✅ Imported {imported_count} summaries")
        return imported_count

    def clean_old_summaries(self, days=30):
        """Clean up summaries older than specified days"""
        print(f"🧹 Cleaning summaries older than {days} days...")

        cutoff_date = datetime.now() - timedelta(days=days)

        old_summaries = (
            self.db.query(ContentSummary)
            .filter(ContentSummary.created_at < cutoff_date)
            .all()
        )

        for summary in old_summaries:
            self.db.delete(summary)

        self.db.commit()
        print(f"✅ Cleaned {len(old_summaries)} old summaries")
        return len(old_summaries)


async def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()
    manager = TrendingSummaryManager()

    try:
        if command == "check":
            missing, total = manager.check_missing_summaries()

        elif command == "generate":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            generated = await manager.generate_missing_summaries(limit)

        elif command == "export":
            filename = sys.argv[2] if len(sys.argv) > 2 else "trending_summaries.json"
            exported = manager.export_summaries(filename)

        elif command == "import":
            if len(sys.argv) < 3:
                print("❌ Error: Please specify the JSON file to import")
                return
            filename = sys.argv[2]
            imported = manager.import_summaries(filename)

        elif command == "clean":
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            cleaned = manager.clean_old_summaries(days)

        else:
            print(f"❌ Unknown command: {command}")
            print(__doc__)

    finally:
        manager.close()


if __name__ == "__main__":
    asyncio.run(main())