#!/usr/bin/env python3
"""
Fix broken wikilinks in Morpheus Vault for MkDocs compatibility.

Problem: Obsidian uses shortest-path resolution for [[Name]], but MkDocs + roamlinks
needs either exact filenames or relative paths.

Solution: Replace [[short-name]] with [[actual/relative/path/file|display-name]]
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict

VAULT_ROOT = Path("/Users/peterzhang/project/morpheus-vault")
# The docs directory determines what MkDocs can serve
DOCS_DIR = VAULT_ROOT / "docs"

# Directories served by MkDocs (via symlinks in docs/)
DOCS_SYMLINKS = {}
for item in DOCS_DIR.iterdir():
    if item.is_symlink() or item.is_dir():
        real = item.resolve()
        DOCS_SYMLINKS[item.name] = real

# Regex for wikilinks: [[target]] or [[target|alias]]
# Also handles ![[embeds]]
WIKILINK_RE = re.compile(r'(!?)\[\[([^\[\]]+?)\]\]')

# Pattern to detect numeric/coordinate-like false positives
NUMERIC_RE = re.compile(r'^[\d,.\s\-+eE]+$')

def is_false_positive(name: str) -> bool:
    """Filter out things that look like wikilinks but aren't."""
    name = name.strip()
    # Pure numbers
    if NUMERIC_RE.match(name):
        return True
    # URLs
    if name.startswith(('http://', 'https://', 'ftp://')):
        return True
    # Anchors only
    if name.startswith('#'):
        return True
    return False

def build_file_index(vault_root: Path):
    """
    Build mappings:
    - stem_to_paths: filename_stem -> [list of relative paths from vault root]
    - name_to_paths: filename_with_ext -> [list of relative paths]
    - relative_path_set: set of all relative paths (without .md extension too)
    """
    stem_to_paths = defaultdict(list)
    name_to_paths = defaultdict(list)
    relative_paths = set()
    relative_paths_no_ext = set()

    for md_file in vault_root.rglob("*.md"):
        # Skip hidden dirs, site dir, node_modules
        rel = md_file.relative_to(vault_root)
        parts = rel.parts
        if any(p.startswith('.') for p in parts):
            continue
        if 'site' in parts or 'node_modules' in parts:
            continue

        rel_str = str(rel)
        rel_no_ext = str(rel.with_suffix(''))

        relative_paths.add(rel_str)
        relative_paths_no_ext.add(rel_no_ext)

        stem = md_file.stem
        stem_to_paths[stem].append(rel_str)
        name_to_paths[md_file.name].append(rel_str)

    return stem_to_paths, name_to_paths, relative_paths, relative_paths_no_ext

def is_in_docs(rel_path: str) -> bool:
    """Check if a relative path is served by MkDocs (in docs/ symlinks)."""
    parts = Path(rel_path).parts
    if not parts:
        return False
    top_dir = parts[0]
    # Check if top-level dir is symlinked in docs/
    return top_dir in DOCS_SYMLINKS or (DOCS_DIR / rel_path).exists()

def resolve_wikilink(target: str, source_file: Path, stem_to_paths, name_to_paths, relative_paths, relative_paths_no_ext):
    """
    Try to resolve a wikilink target to an actual file path.
    Returns (resolved_path, status) where status is 'ok', 'fixed', or 'dead'.
    
    'ok' = already resolvable (exact match or in same dir)
    'fixed' = found a match via stem lookup
    'dead' = no match found
    """
    # Split off any anchor
    anchor = ''
    if '#' in target:
        target_part, anchor = target.split('#', 1)
        anchor = '#' + anchor
        target = target_part

    if not target:
        # Pure anchor link
        return None, 'ok'

    # Check if target already contains path separators (already a path)
    # If it resolves directly, it's fine
    if target in relative_paths or target in relative_paths_no_ext:
        return None, 'ok'
    if target + '.md' in relative_paths:
        return None, 'ok'

    # Check relative to source file's directory
    source_dir = source_file.parent.relative_to(VAULT_ROOT)
    rel_target = str(source_dir / target)
    if rel_target in relative_paths or rel_target in relative_paths_no_ext:
        return None, 'ok'
    if rel_target + '.md' in relative_paths:
        return None, 'ok'

    # Check if target has an extension (could be an image or other file)
    if '.' in Path(target).name:
        # Could be image embed like ![[image.png]]
        # Try to find it
        ext = Path(target).suffix
        if ext.lower() in ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.pdf', '.mp3', '.mp4'):
            return None, 'skip'  # Don't fix media embeds
        # Try as filename with extension
        if target in [Path(p).name for p in relative_paths]:
            # Find the full path
            for p in relative_paths:
                if Path(p).name == target:
                    return p + anchor, 'fixed'

    # Try stem-based lookup
    stem = Path(target).stem if '/' not in target else target
    if stem in stem_to_paths:
        candidates = stem_to_paths[stem]
        if len(candidates) == 1:
            # Unambiguous match
            path = candidates[0]
            # Remove .md extension for the link (roamlinks handles it)
            path_no_ext = str(Path(path).with_suffix(''))
            return path_no_ext + anchor, 'fixed'
        else:
            # Multiple matches - try to pick the best one
            # Prefer files in docs-served directories
            docs_candidates = [c for c in candidates if is_in_docs(c)]
            if len(docs_candidates) == 1:
                path_no_ext = str(Path(docs_candidates[0]).with_suffix(''))
                return path_no_ext + anchor, 'fixed'
            # Prefer file where stem == parent dir name (canonical location)
            canonical = [c for c in candidates if Path(c).stem == Path(c).parent.name]
            if len(canonical) == 1:
                path_no_ext = str(Path(canonical[0]).with_suffix(''))
                return path_no_ext + anchor, 'fixed'
            # If still ambiguous, pick the shortest path
            if docs_candidates:
                best = min(docs_candidates, key=len)
            else:
                best = min(candidates, key=len)
            path_no_ext = str(Path(best).with_suffix(''))
            return path_no_ext + anchor, 'fixed'

    # Dead link
    return None, 'dead'

def process_file(md_file: Path, stem_to_paths, name_to_paths, relative_paths, relative_paths_no_ext, dry_run=False):
    """Process a single markdown file, fixing broken wikilinks."""
    content = md_file.read_text(encoding='utf-8')
    fixes = []
    dead_links = []
    
    def replace_wikilink(match):
        embed_prefix = match.group(1)  # '!' for embeds, '' otherwise
        inner = match.group(2)
        
        # Split target and alias
        if '|' in inner:
            target, alias = inner.split('|', 1)
        else:
            target = inner
            alias = None
        
        target = target.strip()
        
        # Skip false positives
        if is_false_positive(target):
            return match.group(0)
        
        # Skip if it looks like a heading-only link
        if not target or target.startswith('#'):
            return match.group(0)
        
        resolved, status = resolve_wikilink(
            target, md_file, stem_to_paths, name_to_paths, 
            relative_paths, relative_paths_no_ext
        )
        
        if status == 'ok' or status == 'skip':
            return match.group(0)
        
        if status == 'dead':
            dead_links.append(target)
            return match.group(0)
        
        if status == 'fixed':
            # Build the replacement
            if alias:
                display = alias
            else:
                # Use the original target as display name (short name)
                # But strip any anchor
                display = target.split('#')[0] if '#' in target else target
            
            new_link = f"{embed_prefix}[[{resolved}|{display}]]"
            fixes.append((target, resolved, display))
            return new_link
        
        return match.group(0)
    
    # Process content, but skip code blocks and inline code
    # Split into segments: code blocks, inline code, and regular text
    new_content = process_content_skip_code(content, replace_wikilink)
    
    if fixes and not dry_run:
        md_file.write_text(new_content, encoding='utf-8')
    
    return fixes, dead_links, new_content != content

def process_content_skip_code(content: str, replacer) -> str:
    """Apply wikilink replacer while skipping code blocks and inline code."""
    # Pattern to match fenced code blocks, inline code, and regular text
    # Order matters: check fenced blocks first, then inline code
    CODE_BLOCK_RE = re.compile(
        r'(```[\s\S]*?```|`[^`\n]+`)',
        re.MULTILINE
    )
    
    parts = CODE_BLOCK_RE.split(content)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # This is a code block or inline code, don't touch it
            result.append(part)
        else:
            # Regular text, apply replacer
            result.append(WIKILINK_RE.sub(replacer, part))
    
    return ''.join(result)


def main():
    dry_run = '--dry-run' in sys.argv
    
    print("=" * 60)
    print("Morpheus Vault Wikilinks Fixer")
    print("=" * 60)
    
    if dry_run:
        print(">>> DRY RUN MODE - no files will be modified <<<\n")
    
    # Step 1: Build file index
    print("Building file index...")
    stem_to_paths, name_to_paths, relative_paths, relative_paths_no_ext = build_file_index(VAULT_ROOT)
    print(f"  Found {len(relative_paths)} markdown files")
    
    # Show ambiguous stems (multiple files with same name)
    ambiguous = {k: v for k, v in stem_to_paths.items() if len(v) > 1}
    if ambiguous:
        print(f"  {len(ambiguous)} ambiguous stems (multiple files with same name):")
        for stem, paths in sorted(ambiguous.items()):
            print(f"    {stem}: {paths}")
    
    # Step 2: Process all markdown files
    print("\nProcessing files...")
    total_fixes = 0
    total_dead = 0
    all_dead_links = defaultdict(list)  # dead_target -> [source_files]
    all_fixes_detail = []
    files_modified = 0
    
    for md_file in sorted(VAULT_ROOT.rglob("*.md")):
        rel = md_file.relative_to(VAULT_ROOT)
        parts = rel.parts
        if any(p.startswith('.') for p in parts):
            continue
        if 'site' in parts or 'node_modules' in parts:
            continue
        
        fixes, dead_links, modified = process_file(
            md_file, stem_to_paths, name_to_paths,
            relative_paths, relative_paths_no_ext, dry_run=dry_run
        )
        
        if fixes:
            files_modified += 1
            for target, resolved, display in fixes:
                total_fixes += 1
                all_fixes_detail.append((str(rel), target, resolved, display))
        
        if dead_links:
            for dl in dead_links:
                total_dead += 1
                all_dead_links[dl].append(str(rel))
    
    # Step 3: Print report
    print("\n" + "=" * 60)
    print("REPAIR REPORT")
    print("=" * 60)
    
    print(f"\n✅ Fixed: {total_fixes} wikilinks in {files_modified} files")
    if all_fixes_detail:
        print("\nFix details:")
        for source, target, resolved, display in all_fixes_detail:
            print(f"  {source}: [[{target}]] → [[{resolved}|{display}]]")
    
    print(f"\n❌ Dead links (no matching file found): {total_dead}")
    if all_dead_links:
        print("\nDead link details:")
        for target, sources in sorted(all_dead_links.items()):
            print(f"  [[{target}]] referenced in:")
            for s in sources:
                print(f"    - {s}")
    
    print("\n" + "=" * 60)
    
    return total_fixes, total_dead

if __name__ == '__main__':
    main()
