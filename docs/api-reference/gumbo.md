# GUMBO API Reference

This page documents the GUMBO proactive-assistant engine (paper §4.3). See the
[GUMBO tutorial](../tutorials/gumbo.md) for a walkthrough.

## Engine

::: gum.gumbo.Gumbo
    handler: python
    selection:
      docstring_style: google
    rendering:
      show_source: true
      show_root_heading: true
      heading_level: 2
      docstring_style: google
      show_root_full_path: true
      show_object_full_path: false
      separate_signature: false
      inherited_members: true

## Suggestion

::: gum.gumbo.Suggestion
    handler: python
    selection:
      docstring_style: google
    rendering:
      show_source: true
      show_root_heading: true
      heading_level: 2
      docstring_style: google
      show_root_full_path: true
      show_object_full_path: false
      separate_signature: false
      inherited_members: true

## Mixed-initiative decision

::: gum.gumbo.expected_utility
    handler: python
    selection:
      docstring_style: google
    rendering:
      show_source: true
      show_root_heading: true
      heading_level: 2
      docstring_style: google
      show_root_full_path: true
      show_object_full_path: false
      separate_signature: false

## De-duplication & rate limiting

::: gum.gumbo.lexical_overlap
    handler: python
    selection:
      docstring_style: google
    rendering:
      show_source: true
      show_root_heading: true
      heading_level: 2
      docstring_style: google
      show_root_full_path: true
      show_object_full_path: false
      separate_signature: false

::: gum.gumbo.TokenBucket
    handler: python
    selection:
      docstring_style: google
    rendering:
      show_source: true
      show_root_heading: true
      heading_level: 2
      docstring_style: google
      show_root_full_path: true
      show_object_full_path: false
      separate_signature: false
      inherited_members: true
