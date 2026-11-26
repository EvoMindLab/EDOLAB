"""Merge general monitoring samples. Scientific run evidence never enters here."""

from .domain import stable_json_loads

FIELDS = ("cpu_percent", "load_1m", "load_5m", "load_15m", "load_ratio",
          "memory_available_bytes", "disk_free_bytes")


def merge_minute(rows):
    """Keep the newest identity/status sample as representative and retain extrema.

    GPU inventories, capacities, and errors describe that representative, rather
    than mixing changing devices. Numeric utilisation fields are weighted means.
    """
    newest = max(rows, key=lambda row: (row["sampled_at"], row["id"]))
    result = dict(newest)
    aggregates = [(row, stable_json_loads(row["aggregation_json"], {})) for row in rows]
    metadata = {"bucket_seconds": 60, "count": sum(agg.get("count", 1) for _, agg in aggregates),
                "start": min(agg.get("start", row["sampled_at"]) for row, agg in aggregates),
                "end": newest["sampled_at"], "gpu_values": "last_sample", "fields": {}}
    for field in FIELDS:
        stats = []
        for row, agg in aggregates:
            if agg:
                saved = agg.get("fields", {}).get(field)
                if saved:
                    stats.append(saved)
            elif row[field] is not None:
                value = row[field]
                stats.append({"count": 1, "sum": value, "min": value, "max": value})
        if stats:
            count = sum(item["count"] for item in stats)
            total = sum(item["sum"] for item in stats)
            metadata["fields"][field] = {"count": count, "sum": total, "min": min(item["min"] for item in stats), "max": max(item["max"] for item in stats)}
            mean = total / count
            result[field] = round(mean) if field.endswith("_bytes") else mean
    return result, metadata
