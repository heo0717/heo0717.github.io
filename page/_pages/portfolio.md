---
layout: archive
title: "architecture"
permalink: /portfolio/
author_profile: true
---

<video width="800" height="600" autoplay loop muted playsinline>
  <source src="/assets/videos/kinetic_grid2.mp4" type="video/mp4">
  브라우저가 영상을 지원하지 않습니다.
</video>

{% for post in site.posts %}
  - [{{ post.title }}]({{ post.url }})
{% endfor %}

