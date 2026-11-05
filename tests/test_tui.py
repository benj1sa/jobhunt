import pytest
from textual.widgets import DataTable, Static

from jobhunt.models.enums import Decision
from jobhunt.models.pipeline import JobScore
from jobhunt.services.queue import QueueItem
from jobhunt.tui import ReviewApp, _bar, render_detail


def test_bar_is_readable_without_color():
    assert _bar(0.0) == "----------"
    assert _bar(0.5) == "#####-----"
    assert _bar(1.4) == "##########"


def test_render_detail_shows_score_breakdown_and_skill_matches(job, requirements):
    score = JobScore(
        job_id=job.id,
        profile_hash="p",
        value=0.8,
        likelihood=0.6,
        priority=0.69,
        components={
            "value": {"interest": {"score": 1.0, "weight": 0.3, "note": "strong alignment"}},
            "likelihood": {"skills": {"score": 0.5, "weight": 0.3, "note": "1/2 required"}},
            "gates": ["active clearance required (x0.2)"],
            "fit": {"detail": {"provider": "openrouter", "model": "m", "rationale": "Good healthcare fit."}},
        },
    )
    extraction_data = requirements.model_dump(mode="json")
    item = QueueItem(
        job=job,
        score=score,
        extraction=None,
        sources=["simplify", "colorstack-board"],
        decision=Decision.LATER,
        review_reason="revisit after finals",
    )
    item.extraction = type("E", (), {"data": extraction_data})()  # lightweight stand-in for the ORM row
    text = render_detail(item, "Full description [with brackets]", have={"python"}, aliases={})
    assert "Priority 0.69" in text
    assert "strong alignment" in text
    assert "Good healthcare fit." in text and "fit by openrouter" in text
    assert "GATE" in text and "clearance" in text
    assert "Python (have)" in text
    assert "simplify, colorstack-board" in text
    assert r"\[with brackets]" in text  # markup is escaped


@pytest.mark.asyncio
async def test_review_app_starts_with_empty_queue():
    app = ReviewApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#queue", DataTable).row_count == 0
        detail = str(app.query_one("#detail", Static).render())
        assert "Queue is empty" in detail
        await pilot.press("r")
        await pilot.pause()
        assert app.show_reviewed is True
